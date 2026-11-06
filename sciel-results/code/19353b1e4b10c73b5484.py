import numpy as np
import time
import random

N = 30
# Precompute triple indices for N=30
i, j, k = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
mask = (i < j) & (j < k)
IDX_I = i[mask].flatten()
IDX_J = j[mask].flatten()
IDX_K = k[mask].flatten()

def fast_evaluate(pts):
    """
    pts: numpy array of shape (30, 2) with integer coordinates
    returns: max multiplicity of non-zero doubled area (int)
    """
    pi = pts[IDX_I]
    pj = pts[IDX_J]
    pk = pts[IDX_K]
    
    # Vectorized determinant calculation
    # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    dets = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    abs_dets = np.abs(dets)
    
    # Exclude zero area
    non_zero = abs_dets[abs_dets > 0]
    
    if len(non_zero) == 0:
        return 0
        
    unique, counts = np.unique(non_zero, return_counts=True)
    return int(np.max(counts))

def generate_random_grid_affine():
    """
    Generates a random affine transformation of a 5x6 grid.
    Returns None if invalid.
    """
    # Base grid 5x6
    cols = np.arange(5)
    rows = np.arange(6)
    G = np.array([(c, r) for c in cols for r in rows])
    G = G.astype(float)
    
    # Random affine transform M
    M = np.array([
        [random.randint(-3, 3), random.randint(-3, 3)],
        [random.randint(-3, 3), random.randint(-3, 3)]
    ])
    
    det = np.linalg.det(M)
    if abs(det) < 1e-6:
        M = np.eye(2)
        
    P = G @ M.T
    
    # Shift to center
    center = P.mean(axis=0)
    P = P - center
    
    # Scale if too big?
    max_coord = np.max(np.abs(P))
    if max_coord > 25:
        scale = 25.0 / max_coord
        P = P * scale
        
    # Round to integers
    P = np.round(P).astype(int)
        
    # Check bounds
    if np.any(P < -30) or np.any(P > 30):
        return None
        
    # Ensure distinct points
    pts_list = [tuple(p) for p in P]
    if len(set(pts_list)) < N:
        return None
        
    return P

def generate_random_uniform():
    """
    Generates 30 distinct random integer points in [-30, 30].
    """
    points = set()
    while len(points) < N:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        points.add((x, y))
    return np.array(list(points))

def local_search(initial_points, deadline):
    """
    Performs local search starting from initial_points.
    Returns best_points found in this run.
    """
    current_points = initial_points.copy()
    current_score = fast_evaluate(current_points)
    best_local_points = current_points.copy()
    best_local_score = current_score
    
    stagnation = 0
    max_stagnation = 500 # Restart local search if stuck
    
    while time.time() < deadline:
        if stagnation > max_stagnation:
            break
            
        # Pick a random index to move
        idx = random.randint(0, N - 1)
        
        # Pick a random new position
        if random.random() < 0.5:
            # Local move
            new_x = current_points[idx][0] + random.randint(-5, 5)
            new_y = current_points[idx][1] + random.randint(-5, 5)
            # Clamp to bounds
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        else:
            # Global move
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        new_point = np.array([new_x, new_y])
        
        # Check if new point is already occupied by another point
        # Exclude the point at idx
        occupied = set(map(tuple, current_points))
        occupied.discard(tuple(current_points[idx]))
        
        if tuple(new_point) in occupied:
            continue
            
        # Swap
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        
        new_score = fast_evaluate(current_points)
        
        if new_score > current_score:
            current_score = new_score
            stagnation = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_points = current_points.copy()
        else:
            # Revert
            current_points[idx] = old_point
            stagnation += 1
            
    return best_local_points, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 # Safety margin
    
    best_score = -1
    best_points = None
    
    # Initial best from seed
    seed_points = np.array([(x, y) for x in range(6) for y in range(5)])
    seed_score = fast_evaluate(seed_points)
    if seed_score > best_score:
        best_score = seed_score
        best_points = seed_points.copy()
        
    while time.time() < deadline:
        # Generate initial configuration
        if random.random() < 0.5:
            init_pts = generate_random_grid_affine()
            if init_pts is None:
                init_pts = generate_random_uniform()
        else:
            init_pts = generate_random_uniform()
            
        # Local search
        local_pts, local_score = local_search(init_pts, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_pts.copy()
            
    # Convert best_points to list of tuples
    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
        
    return [tuple(p) for p in best_points]
