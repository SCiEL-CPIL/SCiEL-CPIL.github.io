"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Find the optimal packing of 100 equal spheres in a unit cube.
    Uses a smooth log-sum-exp objective with annealing, then a final polish.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s hard limit

    # Initialize with a simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()

    def compute_radius(x):
        """Compute the radius of the packing."""
        x = x.reshape(-1, 3)
        # Minimum distance between centers
        from scipy.spatial.distance import cdist
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        # Minimum distance to boundary
        boundary_dist = 0.5 - np.max(np.abs(x), axis=1)
        min_boundary = np.min(boundary_dist)
        return min(min_dist, min_boundary) / 2.0

    def smooth_objective(x, alpha):
        """
        Smooth objective using log-sum-exp.
        We want to maximize the minimum of all pairwise distances and boundary distances.
        This is equivalent to minimizing the log-sum-exp of the negative distances.
        """
        x = x.reshape(-1, 3)
        n = x.shape[0]
        
        # Pairwise distances
        from scipy.spatial.distance import cdist
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        boundary = 0.5 - np.abs(x)  # shape (n, 3)
        boundary_min_per_point = np.min(boundary, axis=1)  # shape (n,)
        
        # Collect all distances
        # Pairwise: all i < j
        iu = np.triu_indices(n, k=1)
        pair_dists = d[iu]
        
        # All distances to consider
        all_dists = np.concatenate([pair_dists, boundary_min_per_point])
        
        # Soft-min via log-sum-exp: -log(sum(exp(-alpha * d))) / alpha
        # This approximates min(d) as alpha -> inf
        # For numerical stability, subtract max
        neg_alpha_d = -alpha * all_dists
        neg_alpha_d -= np.max(neg_alpha_d)
        log_sum = np.log(np.sum(np.exp(neg_alpha_d)))
        soft_min = -(log_sum) / alpha
        
        return soft_min

    def objective_and_grad(x, alpha):
        """Objective and gradient for smooth optimization."""
        x = x.reshape(-1, 3)
        n = x.shape[0]
        
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (n, n, 3)
        d = np.sqrt(np.sum(diff**2, axis=2))  # (n, n)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        boundary = 0.5 - np.abs(x)  # (n, 3)
        boundary_min_idx = np.argmin(boundary, axis=1)  # (n,)
        boundary_min = boundary[np.arange(n), boundary_min_idx]  # (n,)
        
        # Collect all distances and their indices
        iu = np.triu_indices(n, k=1)
        pair_dists = d[iu]
        
        all_dists = np.concatenate([pair_dists, boundary_min])
        
        # Compute weights for soft-min
        neg_alpha_d = -alpha * all_dists
        neg_alpha_d -= np.max(neg_alpha_d)
        weights = np.exp(neg_alpha_d)
        weights /= np.sum(weights)
        
        # Objective value
        obj = -np.sum(weights * all_dists)
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Pairwise contributions
        # For each pair (i, j) with i < j, weight w_ij
        # d_ij = ||x_i - x_j||
        # grad_{x_i} d_ij = (x_i - x_j) / d_ij
        # grad_{x_j} d_ij = (x_j - x_i) / d_ij
        # Contribution to obj gradient: -w_ij * grad d_ij
        
        pair_i, pair_j = iu
        pair_w = weights[:len(pair_dists)]
        
        for k in range(len(pair_dists)):
            i, j = pair_i[k], pair_j[k]
            w = pair_w[k]
            if d[i, j] < 1e-10:
                continue
            diff_ij = x[i] - x[j]
            grad[i] += -w * diff_ij / d[i, j]
            grad[j] += -w * (-diff_ij) / d[i, j]
        
        # Boundary contributions
        boundary_w = weights[len(pair_dists):]
        for k in range(n):
            w = boundary_w[k]
            i = k
            axis = boundary_min_idx[k]
            # boundary_min = 0.5 - |x_i, axis|
            # grad_{x_i, axis} boundary_min = -sign(x_i, axis)
            sign_val = np.sign(x[i, axis])
            if sign_val == 0:
                sign_val = 1.0
            grad[i, axis] += -w * (-sign_val)
        
        return obj, grad

    # Annealing schedule
    alphas = [10, 20, 50, 100, 200, 500, 1000, 2000]
    current_x = x0.copy()
    best_x = x0.copy()
    best_r = compute_radius(x0)
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
        
        # Try to optimize with current alpha
        try:
            result = minimize(
                lambda x: objective_and_grad(x, alpha)[0],
                current_x,
                jac=lambda x: objective_and_grad(x, alpha)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
                current_x = candidate.copy()
        except Exception:
            pass
    
    # Final polish with high precision
    if time.time() < deadline - 2.0:
        # Use a very high alpha for the final polish
        try:
            result = minimize(
                lambda x: objective_and_grad(x, 5000)[0],
                best_x,
                jac=lambda x: objective_and_grad(x, 5000)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e10:
                candidate = result.x.reshape(-1, 3)
                r_candidate = compute_radius(candidate)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube and check
    best_x = np.clip(best_x, -0.5, 0.5)
    
    # Final validation
    final_r = compute_radius(best_x)
    if not np.isfinite(final_r) or final_r <= 0:
        # Fallback to grid
        return grid[:N]
    
    return best_x

# EVOLVE-BLOCK-END
