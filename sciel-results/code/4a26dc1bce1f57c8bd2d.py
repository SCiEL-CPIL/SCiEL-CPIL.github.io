import numpy as np
import time
import random

# Pre-compute triple indices for N=30
N = 30
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)
idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    if points.shape[0] < 3:
        return 0
        
    # Extract coordinates for all triples
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    
    # det = (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    det = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
          (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
          
    valid_dets = np.abs(det)
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    # Count frequencies
    # Max area is bounded. 
    # Use bincount for speed
    counts = np.bincount(valid_dets)
    return counts.max()

def generate_grid_candidates():
    candidates = []
    # Factors of 30
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    for W, H in factors:
        pts = []
        # Center grid
        x_offset = -W // 2
        y_offset = -H // 2
        for i in range(W):
            for j in range(H):
                x = i + x_offset
                y = j + y_offset
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
                else:
                    pts = None
                    break
            if pts is None:
                break
        if pts and len(pts) == N:
            candidates.append(np.array(pts, dtype=int))
            
    # Add some sheared grids (unimodular transforms)
    # A = [[1, k], [0, 1]] or similar
    # Actually, just generate a few random unimodular matrices and apply to the 5x6 grid
    base_grid = None
    for cand in candidates:
        if cand.shape[0] == N:
             # Find the one with W=5, H=6 or similar compact shape
             # For simplicity, just pick the first one that is not 1x30
             if not (min(np.diff(cand[:,0])) == 0 or min(np.diff(cand[:,1])) == 0):
                 base_grid = cand
                 break
    if base_grid is None and candidates:
        base_grid = candidates[0]
        
    if base_grid is not None:
        # Generate 5 random sheared versions
        for _ in range(5):
            # Random unimodular matrix
            # det = 1
            a = random.randint(-2, 2)
            b = random.randint(-2, 2)
            c = random.randint(-2, 2)
            d = random.randint(-2, 2)
            if a * d - b * c != 1:
                continue
            M = np.array([[a, b], [c, d]])
            new_pts = base_grid @ M.T
            # Check bounds and distinctness
            if np.all(new_pts >= -30) and np.all(new_pts <= 30):
                # Check distinctness
                # Since M is unimodular, distinctness is preserved
                candidates.append(new_pts.astype(int))
                
    return candidates

def local_search(initial_pts, deadline):
    current_pts = initial_pts.copy()
    current_score = fast_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = 0.5
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline:
        # Move selection
        r = random.random()
        new_pts = current_pts.copy()
        
        if r < 0.7:
            # 1-Opt Local
            idx = np.random.randint(N)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0: continue
            new_p = current_pts[idx] + np.array([dx, dy])
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30: continue
            # Distinctness
            if np.any(np.all(current_pts != new_p, axis=1) == False):
                 # Check if new_p is in current_pts (excluding idx)
                 # This check is slightly flawed if new_p == current_pts[idx]
                 # But dx,dy != 0,0 so new_p != current_pts[idx]
                 # So we just need to check if new_p is in current_pts
                 if np.any(np.all(current_pts == new_p, axis=1)):
                     continue
            new_pts[idx] = new_p
        else:
            # 1-Opt Global (less frequent)
            idx = np.random.randint(N)
            new_p = np.array([random.randint(-30, 30), random.randint(-30, 30)])
            if np.any(np.all(current_pts == new_p, axis=1)):
                continue
            new_pts[idx] = new_p
            
        new_score = fast_score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            delta = current_score - new_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
            elif random.random() < 0.1: # Accept equal score moves less frequently
                current_pts = new_pts
                current_score = new_score
                
        T = max(T_min, T * alpha)
        
        # Early stopping if no improvement for a long time?
        # Keep it simple, rely on deadline
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    candidates = []
    
    # 1. Grid Candidates
    candidates.extend(generate_grid_candidates())
    
    # 2. Random Candidates
    for _ in range(5):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))
        
    # Evaluate initial candidates
    scored_candidates = []
    for cand in candidates:
        if cand.shape[0] != N: continue
        s = fast_score(cand)
        scored_candidates.append((s, cand))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    
    # Run local search on top candidates
    # We have ~55 seconds.
    # Let's run local search on the top 3 candidates.
    # Time budget per candidate: 15 seconds
    
    num_ls_runs = 3
    time_per_run = 15.0
    
    for i in range(min(num_ls_runs, len(scored_candidates))):
        init_score, init_pts = scored_candidates[i]
        
        if init_score > best_global_score:
            best_global_score = init_score
            best_global_pts = init_pts.copy()
            
        run_deadline = start_time + time_per_run * (i + 1)
        # Ensure we don't exceed global deadline
        if run_deadline > deadline:
            run_deadline = deadline
            
        if time.time() >= run_deadline:
            break
            
        ls_pts, ls_score = local_search(init_pts, run_deadline)
        
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_pts = ls_pts.copy()
            
    # If we have time left, do a final polish on the best found
    if best_global_pts is not None and time.time() < deadline - 2.0:
        # Use remaining time for a final SA run on the best point
        final_deadline = deadline - 1.0
        ls_pts, ls_score = local_search(best_global_pts, final_deadline)
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_pts = ls_pts.copy()

    if best_global_pts is None:
        # Fallback
        return [(0, 0)] * N # Should not happen
        
    return [(int(p[0]), int(p[1])) for p in best_global_pts]
