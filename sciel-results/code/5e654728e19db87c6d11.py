import numpy as np
import time
from scipy.optimize import minimize

N = 300

def calculate_score(pts):
    """Calculate the valid score for the given points."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check boundary
    dists_center = np.linalg.norm(pts, axis=1)
    min_boundary = np.min(1.0 - dists_center)
    if min_boundary <= 0:
        return 0.0

    # Check pairwise distances
    # For N=300, O(N^2) is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dists[iu])
    
    r = min(min_boundary, min_pair_dist / 2.0)
    return r

def objective_logsumexp(x, beta=50.0):
    """
    Objective function to minimize.
    Minimizes a smooth approximation of the negative minimum distance.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    d_ij = dists[iu]
    
    # Boundary distances
    r_bound = 1.0 - np.linalg.norm(pts, axis=1)
    
    # LogSumExp calculation for pairwise distances
    # We minimize logsumexp(-beta * d_ij)
    term_d = -beta * d_ij
    max_term_d = np.max(term_d)
    if np.isfinite(max_term_d):
        sum_exp_d = np.sum(np.exp(term_d - max_term_d))
        logsumexp_d = max_term_d + np.log(sum_exp_d)
    else:
        logsumexp_d = 0.0
        
    # LogSumExp calculation for boundary distances
    term_b = -beta * r_bound
    max_term_b = np.max(term_b)
    if np.isfinite(max_term_b):
        sum_exp_b = np.sum(np.exp(term_b - max_term_b))
        logsumexp_b = max_term_b + np.log(sum_exp_b)
    else:
        logsumexp_b = 0.0
        
    # The value is proportional to -min(distance)
    # Minimizing this maximizes the minimum distance
    return (logsumexp_d + logsumexp_b) / beta

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    best_score = 0.0
    best_points = None
    
    # 1. Generate Initial Hexagonal Lattice
    # Approximate spacing for N circles in unit disk
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    points = []
    k = int(1.0 / s) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # Select N points closest to center to ensure a dense, symmetric start
    if len(points) >= N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        initial_points = points[idx]
    else:
        # Fallback: pad with random points if lattice was too sparse
        while len(points) < N:
            p = np.random.uniform(-1, 1, size=2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        initial_points = points[:N]

    # 2. Optimization Loop with Restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Add noise to escape local minima
        if restart == 0:
            x0 = initial_points.copy()
        else:
            noise_scale = 0.02 * (restart / n_restarts) # Increase noise slightly
            x0 = initial_points + np.random.normal(0, noise_scale, size=(N, 2))
            
        # Ensure initial points are roughly inside
        # L-BFGS-B will handle it, but starting outside is bad
        norms = np.linalg.norm(x0, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            x0[mask] = x0[mask] / norms[mask, np.newaxis] * 0.99

        try:
            res = minimize(
                objective_logsumexp, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.fun < np.inf:
                pts = res.x.reshape(-1, 2)
                
                # Post-processing: Project points back into the circle if they drifted out
                norms = np.linalg.norm(pts, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / norms[mask, np.newaxis] * (1.0 - 1e-9)
                
                score = calculate_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
                    
        except Exception:
            continue

    if best_points is None:
        # Fallback to initial lattice if optimization failed
        return initial_points

    return best_points
