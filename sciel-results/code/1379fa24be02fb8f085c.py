import numpy as np
import time
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF_SIDE = 0.5

def compute_radius(centers):
    """
    Compute the radius of the largest circle packing for given centers.
    centers: (N, 2) array
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check boundaries
    # Distance to each boundary
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    # dist to left: x - (-0.5) = x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    
    # Vectorized boundary distance
    # centers shape (N, 2)
    dists_to_boundaries = np.min(np.stack([
        centers[:, 0] + HALF_SIDE,
        HALF_SIDE - centers[:, 0],
        centers[:, 1] + HALF_SIDE,
        HALF_SIDE - centers[:, 1]
    ], axis=1), axis=1)
    
    min_boundary_dist = np.min(dists_to_boundaries)
    
    # Check pairwise distances
    # Calculate all pairwise distances
    # dist[i, j] = || centers[i] - centers[j] ||
    # We only need min for i < j
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Find minimum pairwise distance
    # Mask diagonal and upper triangle? 
    # Easier: just take min of all off-diagonal
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Radius is half the min distance
    r = min(min_boundary_dist, min_pair_dist / 2.0)
    
    # Validity check: if any center is outside, radius is effectively 0 or negative?
    # The problem says "A centre on or outside the square ... make the candidate invalid."
    # However, the score formula is min(...). If a center is outside, min_boundary_dist < 0.
    # If we return negative radius, it's worse than 0.
    # But the evaluator might treat invalid as 0.
    # To be safe, if any dist is negative, return 0? Or let the min be negative?
    # "Score: the largest common radius r = ... for which the circles are disjoint and inside the square"
    # If not inside, it's invalid.
    # Let's return 0 if any boundary dist is < 0.
    if min_boundary_dist < 0:
        return 0.0
        
    return r

def objective(params):
    """
    Objective function to minimize.
    We want to maximize radius, so minimize -radius.
    """
    centers = params.reshape(N, 2)
    r = compute_radius(centers)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Generate initial guesses
    inits = []
    
    # 1. Grid 7x7 with 2 extra? 
    # 50 = 5 * 10 or 7 * 7 + 1?
    # Let's try 5x10 grid
    m1, m2 = 5, 10
    ticks_x = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks_y = (np.arange(m2) + 0.5) / m2 - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    inits.append(grid)
    
    # 2. Grid 10x5
    m1, m2 = 10, 5
    ticks_x = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks_y = (np.arange(m2) + 0.5) / m2 - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    inits.append(grid)
    
    # 3. Jittered 7x7 (49) + 1? 
    # Let's just use random jitter on a 5x10 grid
    base = inits[0].copy()
    np.random.seed(42)
    jitter = np.random.uniform(-0.05, 0.05, size=(N, 2))
    inits.append(base + jitter)
    
    # 4. Hexagonal-like packing approximation
    # This is harder to generate perfectly for 50 in a square.
    # Let's just use random uniform as a last resort, though it's poor.
    np.random.seed(123)
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
    
    # Optimization loop
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B for efficiency
            res = minimize(
                objective, 
                init.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 0: # If it found something
                centers = res.x.reshape(N, 2)
                r = compute_radius(centers)
                
                if r > best_radius:
                    best_radius = r
                    best_centers = centers
                    
        except Exception as e:
            pass
    
    # If we still have time, do a polish on the best result
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            # Polish with higher precision
            res = minimize(
                objective, 
                best_centers.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            centers = res.x.reshape(N, 2)
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers
        except:
            pass
            
    if best_centers is None:
        # Fallback to grid
        return inits[0]
        
    return best_centers
