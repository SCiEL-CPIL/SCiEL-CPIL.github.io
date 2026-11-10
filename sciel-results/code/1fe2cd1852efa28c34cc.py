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
    # p1, p2, p3 shape (N, 1, 1, 2), (1, N, 1, 2), (1, 1, N, 2)
    p1 = points[:, None, None, :]
    p2 = points[None, :, None, :]
    p3 = points[None, None, :, :]
    
    # v1 = p2 - p1, v2 = p3 - p1
    # det = v1_x * v2_y - v1_y * v2_x
    det = (p2[..., 0] - p1[..., 0]) * (p3[..., 1] - p1[..., 1]) - \
          (p2[..., 1] - p1[..., 1]) * (p3[..., 0] - p1[..., 0])
          
    # We only care about i < j < k
    idx = np.indices((n, n, n))
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    
    valid_dets = np.abs(det[mask])
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    # Count frequencies of each area
    # Max area is bounded by coordinate range.
    # Coordinates in [-30, 30], so max diff is 60.
    # Max det ~ 60*60 = 3600.
    # bincount is efficient for small integer ranges.
    counts = np.bincount(valid_dets)
    return counts.max()

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Initialize with a few different structures and pick the best
    candidates = []
    
    # 1. Grid
    # 6x5 grid
    grid_pts = [(x, y) for x in range(6) for y in range(5)]
    # Shift to center? Not strictly necessary, but helps with symmetry maybe.
    # The problem says [-30, 30].
    # Let's center it: x in -2..3, y in -2..2
    grid_pts = [(x-2, y-2) for x in range(6) for y in range(5)]
    candidates.append(np.array(grid_pts, dtype=int))
    
    # 2. Circle
    # Points on a circle of radius R
    # 30 points
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    R = 30
    circle_pts = np.array([[int(np.cos(a)*R), int(np.sin(a)*R)] for a in angles])
    # Ensure distinct
    # Circle points might collide if R is small or angles close.
    # With R=30, cos/sin values are distinct enough for 30 points.
    # Check distinctness
    if len(set(map(tuple, circle_pts))) == N:
        candidates.append(circle_pts)
    else:
        # Fallback to random if collision
        pass
        
    # 3. Random
    random_pts = np.random.randint(-30, 31, size=(N, 2))
    # Ensure distinct
    # Simple check
    if len(set(map(tuple, random_pts))) == N:
        candidates.append(random_pts)
    else:
        # Force distinct
        pts = []
        while len(pts) < N:
            p = tuple(np.random.randint(-30, 31, size=2))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Evaluate candidates
    best_pts = None
    best_score = 0
    for cand in candidates:
        s = fast_score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand.copy()
            
    # Simulated Annealing
    current_pts = best_pts
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random index
        idx = np.random.randint(N)
        
        # Generate new point
        new_p = np.array([np.random.randint(-30, 31), np.random.randint(-30, 31)])
        
        # Check distinctness
        # If new_p is already in current_pts (except at idx), reject or retry
        # For efficiency, just check if it's in the set of other points
        # Since N is small, a simple loop or set check is fine.
        # However, creating a set every iteration is overhead.
        # Let's just try it and check score validity? No, distinctness is a hard constraint.
        # If not distinct, the problem is invalid.
        # Let's ensure distinctness.
        is_distinct = True
        for i in range(N):
            if i != idx and np.array_equal(current_pts[i], new_p):
                is_distinct = False
                break
        
        if not is_distinct:
            T *= cooling
            continue
            
        # Calculate new score
        new_pts = current_pts.copy()
        new_pts[idx] = new_p
        
        new_score = fast_score(new_pts)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta > 0:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            # Note: score is integer. T is float.
            # If delta is large negative, prob is 0.
            if random.random() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
                
        T *= cooling
        
    return [(int(p[0]), int(p[1])) for p in best_pts]

# EVOLVE-BLOCK-END
