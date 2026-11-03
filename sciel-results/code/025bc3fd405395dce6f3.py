import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

# Ring structure: 1 center, then rings of 6, 12, 18, 13
RING_COUNTS = [1, 6, 12, 18, 13]
RING_RADII_INIT = [0.0, 0.25, 0.5, 0.75, 0.95]
RING_OFFSETS_INIT = [0.0, 0.0, 0.0, 0.0, 0.0]

def _compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def _params_to_points(params):
    """
    Convert 11 parameters to 50 points.
    params[0]: center x
    params[1]: center y
    params[2:7]: radii of rings 1-4 (indices 2,3,4,5)
    params[7:11]: angular offsets of rings 1-4 (indices 7,8,9,10)
    """
    points = np.zeros((N, 2))
    idx = 0
    
    # Center
    points[idx, 0] = params[0]
    points[idx, 1] = params[1]
    idx += 1
    
    # Rings
    for i, count in enumerate(RING_COUNTS[1:]):
        radius = params[2 + i]
        offset = params[7 + i]
        angles = 2 * np.pi * np.arange(count) / count + offset
        points[idx:idx+count, 0] = radius * np.cos(angles)
        points[idx:idx+count, 1] = radius * np.sin(angles)
        idx += count
        
    return points

def _objective(params):
    """Objective for Differential Evolution: minimize -radius."""
    points = _params_to_points(params)
    r = _compute_radius(points)
    return -r

def _local_polish(points, time_limit=10.0):
    """
    Perform local optimization on the free 100D coordinates to refine the ring structure.
    Uses a simple gradient-free or gradient-based method? 
    Given the time limit and complexity, let's use a few iterations of a simple 
    coordinate descent or just rely on the DE result if it's good.
    Actually, a better approach: Use scipy minimize with L-BFGS-B on the free coordinates
    with a soft-min objective, similar to the previous program but starting from a good 
    structured init.
    """
    if time_limit <= 0:
        return points
    
    # Define soft-min objective for L-BFGS-B
    def soft_min_obj(x, beta=100.0):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        f = 1.0 - norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        g = dists / 2.0
        
        all_margins = np.concatenate([f.ravel(), g.ravel()])
        
        # Soft min
        max_margin = np.max(all_margins)
        shifted = beta * (max_margin - all_margins)
        shifted = np.clip(shifted, -500, 500)
        exp_shifted = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp_shifted))
        obj = max_margin - (1.0 / beta) * log_sum_exp
        
        # Gradient
        p = exp_shifted / np.sum(exp_shifted)
        grad = np.zeros_like(x)
        
        # Boundary
        for i in range(N):
            if norms[i] > 1e-10:
                grad[2*i:2*i+2] += p[i] * (-pts[i] / norms[i])
        
        # Pairwise
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w = p[N + i*N + j]
                if w < 1e-15:
                    continue
                d = dists[i, j]
                if d > 1e-10:
                    vec = pts[i] - pts[j]
                    grad[2*i:2*i+2] += w * (vec / (2.0 * d))
                    grad[2*j:2*j+2] += w * (-vec / (2.0 * d))
        
        return -obj, -grad

    start_time = time.time()
    x0 = points.flatten()
    
    try:
        result = minimize(soft_min_obj, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6,
                                   'time_limit': time_limit})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x.reshape(N, 2)
    except Exception:
        pass
        
    return points

def construct():
    deadline = time.time() + DEADLINE
    
    np.random.seed(42)
    
    # Define bounds for the 11 parameters
    # Center x, y: [-0.1, 0.1]
    # Radii: [0.0, 1.0]
    # Offsets: [0, 2*pi]
    bounds = [
        (-0.1, 0.1), # center x
        (-0.1, 0.1), # center y
        (0.1, 0.4),  # ring 1 radius
        (0.3, 0.6),  # ring 2 radius
        (0.5, 0.8),  # ring 3 radius
        (0.7, 0.99), # ring 4 radius
        (0, 2*np.pi), # ring 1 offset
        (0, 2*np.pi), # ring 2 offset
        (0, 2*np.pi), # ring 3 offset
        (0, 2*np.pi), # ring 4 offset
    ]
    
    # Initial guess
    x0 = np.array([0.0, 0.0] + RING_RADII_INIT[1:] + RING_OFFSETS_INIT[1:])
    
    # Run Differential Evolution
    # Use a moderate number of iterations and population to balance time and quality
    # maxiter=100, popsize=15 -> 1500 evaluations
    # Each eval is O(N^2) = 2500 ops, very fast.
    # 1500 * 2500 = 3.75M ops, should take < 1s.
    
    try:
        de_result = differential_evolution(
            _objective, bounds, x0=x0, 
            maxiter=200, popsize=20, 
            tol=1e-6, seed=42,
            polish=False # We will polish manually to control time
        )
    except Exception:
        de_result = None
    
    best_params = de_result.x if de_result is not None else x0
    best_points = _params_to_points(best_params)
    best_r = _compute_radius(best_points)
    
    # Check time
    elapsed = time.time() - (deadline - DEADLINE)
    if elapsed > DEADLINE - 20:
        return best_points
    
    # Local polish on the free coordinates
    time_left = deadline - time.time() - 5
    if time_left > 5:
        polished_points = _local_polish(best_points, time_limit=time_left)
        polished_r = _compute_radius(polished_points)
        
        if polished_r > best_r:
            best_points = polished_points
            best_r = polished_r
    
    # Final validation and boundary enforcement
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.999:
            best_points[i] *= 0.999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = _compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
