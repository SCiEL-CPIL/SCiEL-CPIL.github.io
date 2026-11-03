"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian.
    Point = u*V1 + v*V2 (since V0=0)
    """
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _softmin_objective(x, beta=100.0, penalty_weight=100.0):
    """
    Objective: -softmin(areas) + penalty for out-of-bounds.
    x: flat array of barycentric coords (2*N).
    """
    bary = x.reshape(-1, 2)
    
    # Penalty for u < 0, v < 0, u+v > 1
    # We use a smooth penalty: sum of squares of violations
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Violations
    viol_u = np.clip(-u, 0, None)
    viol_v = np.clip(-v, 0, None)
    viol_sum = np.clip(u + v - 1.0, 0, None)
    
    penalty = penalty_weight * (np.sum(viol_u**2) + np.sum(viol_v**2) + np.sum(viol_sum**2))
    
    # For area calculation, we must use valid points. 
    # If points are out of bounds, the area calculation is still mathematically valid 
    # for the Cartesian projection, but the "inside" constraint is violated.
    # We just compute the area of the projected points.
    # To avoid numerical issues with huge penalties, we can clamp for the area part 
    # but keep the penalty for the optimization direction.
    bary_clamped = bary.copy()
    bary_clamped = np.clip(bary_clamped, 0.0, None)
    s = bary_clamped[:, 0] + bary_clamped[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary_clamped[mask] = bary_clamped[mask] / s[mask, None]
        
    points = _bary_to_cart(bary_clamped)
    
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to MAXIMIZE min(areas), so we MINIMIZE -min(areas).
    # Soft-min approximates min(areas).
    min_area = np.min(areas)
    # Shift for numerical stability
    shifted = areas - min_area
    # log-sum-exp trick
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min + penalty

def _make_initial_bary_grid(seed):
    """Generate a structured grid-based initial barycentric configuration."""
    rng = np.random.RandomState(seed)
    
    # Key structural points: vertices, edge midpoints, centroid
    bary_list = [
        [0.0, 0.0],       # V0
        [1.0, 0.0],       # V1
        [0.0, 1.0],       # V2
        [0.5, 0.0],       # midpoint V0-V1
        [0.5, 0.5],       # midpoint V1-V2
        [0.0, 0.5],       # midpoint V2-V0
        [1.0/3, 1.0/3],   # centroid
    ]
    
    # Add interior points in a regular pattern (hexagonal-like)
    # We need 4 more to reach 11.
    # Points at 1/4 and 3/4 along medians?
    # Or just a fine grid.
    
    # Let's add points at specific barycentric coordinates
    additional = [
        [0.25, 0.25],
        [0.25, 0.5],
        [0.5, 0.25],
        [0.5, 0.0], # Duplicate? No, [0.5, 0.0] is midpoint. 
        # Let's pick distinct ones
    ]
    
    # Better: Generate 11 points on a grid in barycentric space
    # i from 0 to 3, j from 0 to 2?
    # a = i/3, b = j/3?
    
    # Let's just use a jittered version of a known good config or random.
    # The grid init in the inspiration was:
    additional = [
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.75, 0.25],
    ]
    
    for pt in additional:
        if len(bary_list) < N:
            if pt[0] >= 0 and pt[1] >= 0 and pt[0] + pt[1] <= 1:
                bary_list.append(pt)
    
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_list.append([u, v])
    
    return np.array(bary_list[:N])

def _make_initial_bary_random(seed):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(1000):
        u = rng.uniform(0.0, 1.0)
        v = rng.uniform(0.0, 1.0)
        if u + v <= 1.0:
            bary_list.append([u, v])
            if len(bary_list) == N:
                break
    if len(bary_list) < N:
        # Fallback
        for _ in range(N - len(bary_list)):
            u = rng.uniform(0.1, 0.8)
            v = rng.uniform(0.1, 0.8)
            if u + v < 0.9:
                bary_list.append([u, v])
    return np.array(bary_list[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Define initializations
    inits = []
    
    # 1. Structured Grid
    inits.append(_make_initial_bary_grid(42))
    
    # 2. Random Restarts
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        inits.append(_make_initial_bary_random(seed))
        
    # 3. Perturbed Grid
    base_grid = _make_initial_bary_grid(42)
    rng_pert = np.random.RandomState(99)
    for scale in [0.05, 0.1, 0.2]:
        perturbed = base_grid + rng_pert.normal(0, scale, base_grid.shape)
        # Clamp to valid region
        perturbed = np.clip(perturbed, 0, None)
        s = perturbed[:, 0] + perturbed[:, 1]
        mask = s > 1.0
        if np.any(mask):
            perturbed[mask] = perturbed[mask] / s[mask, None]
        inits.append(perturbed)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # We will run L-BFGS-B on each init
    # We use a fixed beta for now, or adaptive? 
    # Fixed beta=100 is usually good enough for 11 points.
    # If we have time, we can try increasing beta in later iterations.
    
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        x0 = init_bary.flatten()
        
        # Try a few different betas? Or just one good one.
        # Let's try beta=50 and beta=200 for the first few inits to be safe.
        betas = [50.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                result = minimize(
                    _softmin_objective,
                    x0,
                    args=(beta, 100.0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
                
            if time.time() > deadline - 5.0:
                break
                
            opt_bary = result.x.reshape(-1, 2)
            
            # Final projection to ensure validity
            opt_bary = np.clip(opt_bary, 0, None)
            s = opt_bary[:, 0] + opt_bary[:, 1]
            mask = s > 1.0
            if np.any(mask):
                opt_bary[mask] = opt_bary[mask] / s[mask, None]
                
            points = _bary_to_cart(opt_bary)
            
            # Verify all points are in triangle (they should be by construction)
            # Calculate true min area
            score = _min_triangle_area_vectorized(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    if best_points is None:
        # Fallback: Return the initial grid points
        best_points = _bary_to_cart(_make_initial_bary_grid(42))
        
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Min Area:", _min_triangle_area_vectorized(pts))
