import numpy as np
import time
import random

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    n = points.shape[0]
    if n < 3:
        return 0
        
    # Vectorized calculation of determinants for all triples (i, j, k)
    p1 = points[:, None, None, :]
    p2 = points[None, :, None, :]
    p3 = points[None, None, :, :]
    
    # det = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    det = (p2[..., 0] - p1[..., 0]) * (p3[..., 1] - p1[..., 1]) - \
          (p2[..., 1] - p1[..., 1]) * (p3[..., 0] - p1[..., 0])
          
    # We only care about i < j < k
    idx = np.indices((n, n, n))
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    
    valid_dets = np.abs(det[mask])
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    # Count frequencies
    # Max area for coords in [-30, 30] is roughly 3600.
    # Using bincount is efficient.
    counts = np.bincount(valid_dets)
    return counts.max()

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Rectangular Grids W x H such that W*H = 30
    for W in [1, 2, 3, 5, 6, 10, 15, 30]:
        H = 30 // W
        if W * H != 30: continue
        # Center the grid in [-30, 30]
        # x from -W//2 to W//2 - 1 (adjust for even/odd)
        # Simpler: generate 0..W-1, 0..H-1, then shift
        pts = []
        for i in range(W):
            for j in range(H):
                # Shift to center around 0
                # If W is even, range is -W/2 .. W/2 - 1
                # If W is odd, range is -W//2 .. W//2
                x = i - W // 2
                y = j - H // 2
                pts.append((x, y))
        
        # Check bounds
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(np.array(pts, dtype=int))
            
    # Also add a few random starts
    for _ in range(5):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Evaluate candidates
    best_pts = None
    best_score = -1
    for cand in candidates:
        s = fast_score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()
            
    current_pts = best_pts
    current_score = best_score
    
    # 2. Local Search (Hill Climbing with Random Jumps)
    # We alternate between small local moves and large random jumps to escape local optima.
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Decide move type: 80% local, 20% global
        if random.random() < 0.8:
            # Local move: shift by -2 to 2 in x and y
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_p = current_pts[idx] + np.array([dx, dy])
        else:
            # Global move: random point in box
            new_p = np.array([random.randint(-30, 30), random.randint(-30, 30)])
            
        # Check bounds
        if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
            continue
            
        # Check distinctness
        # If new_p is already in current_pts (except at idx), reject
        is_distinct = True
        for i in range(N):
            if i != idx and np.array_equal(current_pts[i], new_p):
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        # Calculate new score
        new_pts = current_pts.copy()
        new_pts[idx] = new_p
        
        new_score = fast_score(new_pts)
        
        # Accept if better
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
        # Small chance to accept equal or worse to escape plateaus?
        # Since we want max multiplicity, and it's discrete, 
        # accepting worse moves can help escape local maxima.
        # Let's try a simple simulated annealing accept for worse moves
        # Probability = exp(-delta / T)
        # But T needs to be managed. 
        # Given time constraint, simple hill climbing might get stuck.
        # Let's add a small probability of accepting a worse move.
        elif new_score < current_score and random.random() < 0.05:
             # Accept 5% of the time if worse, to explore
             current_pts = new_pts
             current_score = new_score

    return [(int(p[0]), int(p[1])) for p in best_pts]
