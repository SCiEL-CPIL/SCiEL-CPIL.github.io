"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _soft_min_obj(x, n, beta):
    """Soft-min objective for optimization. x is flattened centers."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    # Soft min using log-sum-exp for stability
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, shift by max
    m = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val

def _smart_init(n):
    """Create a smart initial configuration: 7x7 grid + optimal 50th point."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # Find best spot for the 50th point
    # Sample a fine grid
    sample_res = 100
    xs = np.linspace(-0.5, 0.5, sample_res)
    ys = np.linspace(-0.5, 0.5, sample_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Distance to existing grid points
    diff = pts[:, np.newaxis, :] - grid_49[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    # Distance to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    d_b = np.minimum(d_x, d_y)
    
    # Maximize the minimum clearance
    clearance = np.minimum(min_dists, d_b)
    idx = np.argmax(clearance)
    new_pt = pts[idx]
    
    centers = np.vstack([grid_49, new_pt])
    
    # Small perturbation to break symmetry
    np.random.seed(42)
    centers = centers + np.random.normal(0, 0.005, size=centers.shape)
    
    return centers[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    initial_configs = [
        _smart_init(N),
        _smart_init(N) + np.random.normal(0, 0.02, size=(N, 2)),
    ]
    
    for init_centers in initial_configs:
        if time.time() > deadline - 10.0:
            break
        
        centers = init_centers.copy()
        r = _compute_radius(centers)
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Optimization loop with increasing beta
        # Start with moderate beta, increase to sharpen the minimum
        betas = [20, 50, 100, 200]
        
        current_centers = centers.copy()
        current_r = r
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            x0 = current_centers.flatten()
            
            try:
                result = minimize(
                    _soft_min_obj,
                    x0,
                    args=(N, beta),
                    method='L-BFGS-B',
                    bounds=[(-0.49, 0.49)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                refined_centers = result.x.reshape(N, 2)
                refined_r = _compute_radius(refined_centers)
                
                if refined_r > current_r:
                    current_r = refined_r
                    current_centers = refined_centers
            except Exception:
                pass
        
        if current_r > best_r:
            best_r = current_r
            best_centers = current_centers.copy()
    
    # Final polish with high beta
    if time.time() < deadline - 2.0 and best_centers is not None:
        try:
            x0 = best_centers.flatten()
            result = minimize(
                _soft_min_obj,
                x0,
                args=(N, 500),
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            refined_centers = result.x.reshape(N, 2)
            refined_r = _compute_radius(refined_centers)
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined_centers
        except Exception:
            pass

    # Ensure validity
    if best_centers is not None:
        eps = 1e-9
        best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    else:
        best_centers = _smart_init(N)
        best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
