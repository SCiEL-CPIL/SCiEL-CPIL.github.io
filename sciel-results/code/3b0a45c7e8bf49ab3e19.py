import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _score(centers):
    """Calculate the radius (score) for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    # Efficiently compute min distance
    # For N=200, N^2 is 40000, which is fine.
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distances
    dists_boundary = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_boundary = np.min(dists_boundary)
    
    return min(min_dist, min_boundary) / 2.0

def _objective(x, N):
    """
    Objective function to minimize.
    We minimize the sum of inverse squared distances to spread circles.
    x is flattened array of shape (2N,).
    """
    centers = x.reshape(N, 2)
    
    # Calculate pairwise distances
    # Use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_sq = dists_sq[mask]
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # Sum of inverse squared distances
    val = np.sum(1.0 / dists_sq)
    
    return val

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_score = _score(best_centers)
    
    # Precompute indices for objective? No, L-BFGS-B calls it many times.
    # We can speed up objective by precomputing indices, but N=200 is small enough.
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Multi-start local search
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() - start_time < deadline:
        restarts += 1
        
        # Initialize: Grid with perturbation
        # Perturbation size decreases with restarts? Or just random.
        # Let's try random initialization near the grid or purely random.
        # Purely random often gets stuck in bad local minima.
        # Grid + perturbation is safer.
        
        # Perturbation magnitude
        # Early restarts: larger perturbation to explore.
        # Later restarts: smaller.
        pert_scale = 0.1 / (1 + restarts * 0.1)
        x0 = best_centers + rng.uniform(-pert_scale, pert_scale, size=(N, 2))
        x0 = np.clip(x0, -0.49, 0.49) # Keep safely inside
        
        # Run optimization
        try:
            res = minimize(
                _objective, 
                x0.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
        except Exception:
            continue
            
        x_opt = res.x.reshape(N, 2)
        x_opt = np.clip(x_opt, -0.5, 0.5)
        
        score = _score(x_opt)
        if score > best_score:
            best_score = score
            best_centers = x_opt.copy()
            
        # Check time
        if time.time() - start_time > deadline:
            break
            
    return best_centers
