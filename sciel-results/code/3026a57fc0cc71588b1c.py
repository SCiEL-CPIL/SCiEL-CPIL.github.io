import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50

def compute_radius(points):
    """
    Compute the radius of the largest circle packing for given points.
    points: (N, 2) array
    returns: r
    """
    if len(points) < 2:
        return 0.0
    
    # Pairwise distances
    # points[:, np.newaxis] - points[np.newaxis, :]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Minimum distance between any two distinct points
    # Mask out diagonal
    mask = ~np.eye(len(points), dtype=bool)
    min_dist = np.min(dist[mask])
    
    # Distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def smooth_min(values, beta):
    """
    Smooth approximation of min(values).
    """
    # To avoid overflow in exp, subtract max
    m = np.max(values)
    if not np.isfinite(m):
        return -np.inf
    # min(x) = - max(-x)
    # -1/beta * log(sum(exp(-beta * x)))
    # Let y = -beta * x
    # max(y) = -beta * min(x)
    # log-sum-exp(y) = max(y) + log(sum(exp(y - max(y))))
    # result = -1/beta * (max(y) + log(sum(exp(y - max(y)))))
    
    y = -beta * values
    m_y = np.max(y)
    if not np.isfinite(m_y):
        return -np.inf
    log_sum_exp = m_y + np.log(np.sum(np.exp(y - m_y)))
    return -log_sum_exp / beta

def objective_de(x):
    """
    Objective for Differential Evolution.
    Minimize negative radius.
    """
    points = x.reshape(N, 2)
    # Basic validity check to avoid NaNs in DE
    if not np.all(np.isfinite(points)):
        return 1e10
    r = compute_radius(points)
    return -r

def objective_smooth(x, beta):
    """
    Objective for local polish.
    Minimize negative smooth min of distances and boundary constraints.
    """
    points = x.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # We need min of (distances/2) and (1 - radii)
    # Create a vector of all constraints
    # Distances: N*(N-1)/2 values, divided by 2
    # Boundary: N values
    mask = ~np.eye(N, dtype=bool)
    pair_dists = dist[mask] / 2.0
    
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - radii
    
    all_vals = np.concatenate([pair_dists, boundary_dists])
    
    r_smooth = smooth_min(all_vals, beta)
    return -r_smooth

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin for 120s limit
    
    # Initializations
    # 1. Hexagonal lattice (from seed)
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points_hex = np.array(rows)
    radius_hex = np.hypot(points_hex[:, 0], points_hex[:, 1])
    points_hex = points_hex[radius_hex <= 1.0 - spacing / 2.0]
    if len(points_hex) >= N:
        points_hex = points_hex[np.argsort(radius_hex, kind="stable")[:N]]
    else:
        # Fallback to random if hex fails (unlikely)
        points_hex = np.random.uniform(-0.8, 0.8, (N, 2))
    
    best_points = points_hex.copy()
    best_score = compute_radius(best_points)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return best_points
    
    # 2. Run Differential Evolution
    # Bounds: [-1, 1] for x and y
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # We can try to run DE for a short time
    # DE is stochastic, so we might run it multiple times if time permits, or once with enough generations.
    # Given 100 dimensions, DE is slow.
    # Let's try a smaller population and fewer generations, relying on the polish.
    
    try:
        # maxiter is tricky with time. Let's estimate.
        # One eval ~ 0.5ms? 1500 pop * 50 iters = 75k evals ~ 37s.
        # Let's use popsize=10, maxiter=100.
        # 1000 * 100 = 100k evals.
        
        de_result = differential_evolution(
            objective_de,
            bounds,
            seed=42,
            popsize=10,
            maxiter=100,
            tol=0.01,
            init='sobol', # Better initialization for high dims
            polish=False # We do our own polish
        )
        
        if de_result.success or de_result.fun < -best_score:
            # de_result.fun is negative of radius
            de_radius = -de_result.fun
            if de_radius > best_score:
                best_score = de_radius
                best_points = de_result.x.reshape(N, 2)
                
    except Exception:
        pass # Ignore DE failures
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - 10:
        # Not enough time for polish, return best
        return best_points
    
    # 3. Local Polish
    # Use the smooth objective with a high beta
    # Start with the best points found
    current_points = best_points.flatten()
    
    # Try a few different betas or just one high beta
    betas = [500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        try:
            res = minimize(
                objective_smooth,
                current_points,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < -best_score:
                new_radius = -res.fun
                # Verify with exact min
                new_points = res.x.reshape(N, 2)
                exact_radius = compute_radius(new_points)
                if exact_radius > best_score:
                    best_score = exact_radius
                    best_points = new_points
                    current_points = res.x # Continue from this point for next beta
        except Exception:
            pass

    return best_points
