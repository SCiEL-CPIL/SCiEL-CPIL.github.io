import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Precompute grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    grid_array = np.array(grid_points, dtype=np.int64)
    
    best_points = []
    best_count = 0
    
    # Fast vectorized validity check for local search
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        n = len(P)
        
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        if len(np.unique(d_p)) != n:
            return False
            
        # Squared distances between all pairs in P
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        # Check 2 and 3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
        cond1 = (D_sq == d_p[:, np.newaxis])
        cond2 = (D_sq == d_p[np.newaxis, :])
        
        invalid_mask = cond1 | cond2
        np.fill_diagonal(invalid_mask, False)
        
        if np.any(invalid_mask):
            return False
        
        return True
    
    # Greedy construction with incremental distance tracking
    def run_greedy():
        S = []
        S_array = np.array([]).reshape(0, 2)
        D = np.array([]).reshape(0, 0)
        
        candidates_idx = np.arange(len(grid_points))
        random.shuffle(candidates_idx.tolist())
        
        while len(candidates_idx) > 0 and time.time() < deadline:
            if len(S) == 0:
                idx = candidates_idx[0]
                p = grid_points[idx]
                S.append(p)
                S_array = np.array([p], dtype=np.int64)
                D = np.array([[0]], dtype=np.int64)
                candidates_idx = candidates_idx[1:]
                continue
            
            cand_arr = grid_array[candidates_idx]
            
            diff = cand_arr[:, np.newaxis, :] - S_array[np.newaxis, :, :]
            d_sq = np.sum(diff**2, axis=2)
            
            sorted_d = np.sort(d_sq, axis=1)
            diffs = np.diff(sorted_d, axis=1)
            has_dup = np.any(diffs == 0, axis=1)
            
            conflict_mask = (d_sq[:, :, np.newaxis] == D[np.newaxis, :, :])
            has_conflict = np.any(conflict_mask, axis=(1, 2))
            
            valid_mask = ~(has_dup | has_conflict)
            
            if not np.any(valid_mask):
                break
            
            valid_indices = np.where(valid_mask)[0]
            pick_idx = random.choice(valid_indices.tolist())
            
            p_idx = candidates_idx[pick_idx]
            p = grid_points[p_idx]
            
            S.append(p)
            S_array = np.vstack([S_array, np.array([p])])
            
            new_d = d_sq[pick_idx]
            D_new = np.zeros((len(S), len(S)), dtype=np.int64)
            D_new[:-1, :-1] = D
            D_new[:-1, -1] = new_d
            D_new[-1, :-1] = new_d
            D_new[-1, -1] = 0
            D = D_new
            
            candidates_idx = np.delete(candidates_idx, pick_idx)
        
        return S
    
    # Local search to improve a given set of points
    def local_search(initial_pts):
        pts = list(initial_pts)
        pts_set = set(pts)
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding random points
            for _ in range(2000):
                if time.time() > deadline - 5:
                    break
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p not in pts_set and is_valid_addition(pts, p):
                    pts.append(p)
                    pts_set.add(p)
                    improved = True
            
            # Try swap moves: remove one point, try to add a different one
            if len(pts) > 0:
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    idx = random.randint(0, len(pts) - 1)
                    old_pt = pts[idx]
                    pts_temp = pts[:idx] + pts[idx+1:]
                    pts_temp_set = pts_set - {old_pt}
                    
                    found = False
                    for _ in range(50):
                        if time.time() > deadline - 5:
                            break
                        p = (random.randint(0, N-1), random.randint(0, N-1))
                        if p not in pts_temp_set and is_valid_addition(pts_temp, p):
                            pts = pts_temp + [p]
                            pts_set = pts_temp_set | {p}
                            improved = True
                            found = True
                            break
                    if not found:
                        # Keep original if no improvement
                        pass
        
        return pts
    
    # Run multiple greedy constructions
    for trial in range(200):
        if time.time() > deadline - 15:
            break
        
        pts = run_greedy()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
        
        # Apply local search if we have time and potential
        if len(pts) > best_count - 5 and time.time() < deadline - 20:
            improved_pts = local_search(pts)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
    
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
