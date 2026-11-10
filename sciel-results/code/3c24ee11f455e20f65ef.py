# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses optimization of the sum of log distances (geometric mean) with a scale constraint.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    best_points = None
    best_score = -np.inf

    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def compute_score(points):
        # points shape (n, 2)
        # Compute pairwise distances
        # Using broadcasting for speed
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
        # Mask diagonal
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        d_min = np.min(dists[mask])
        d_max = np.max(dists[mask])
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_and_grad(x):
        # x is flattened points (n*d,)
        points = x.reshape(n, d)
        
        # Center the points (remove translation invariance)
        # Note: We want to optimize the shape, so we center in the objective?
        # If we center in the objective, the gradient w.r.t x is slightly different 
        # because x is not the centered points.
        # Better to enforce centering via initialization and symmetry of forces,
        # OR include a penalty for non-zero centroid.
        # Let's include a penalty for centroid and scale.
        
        centroid = np.mean(points, axis=0)
        norm_sq = np.sum(points**2)
        
        # Target norm_sq = n (average sq norm = 1)
        target_norm_sq = n
        
        # Objective: Maximize sum log dist -> Minimize -sum log dist
        # Add penalties for centroid != 0 and norm_sq != target
        # This makes the problem well-posed.
        
        # Calculate distances
        # Efficient distance calculation
        # dists_sq = np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1)
        # Using the pair indices for just the necessary terms
        diffs = points[i_idx] - points[j_idx]
        dists_sq = np.sum(diffs**2, axis=1)
        # Avoid log(0)
        dists_sq = np.maximum(dists_sq, 1e-12)
        log_dists = 0.5 * np.log(dists_sq)
        sum_log_dists = np.sum(log_dists)
        
        # Penalties
        pen_centroid = np.sum(centroid**2)
        pen_scale = (norm_sq - target_norm_sq)**2
        
        # We want to maximize sum_log_dists, so minimize -sum_log_dists
        # We want to minimize penalties
        val = -sum_log_dists + 100.0 * pen_centroid + 10.0 * pen_scale
        
        # Gradient
        # grad of -sum log dist
        # d/dp_k sum_{j != k} 0.5 log(||p_k - p_j||^2)
        # = sum_{j != k} 0.5 * (1/d^2) * 2(p_k - p_j)
        # = sum_{j != k} (p_k - p_j) / d^2
        
        grad_points = np.zeros_like(points)
        # Accumulate forces
        # For each pair (i, j), force on i is (p_i - p_j)/d^2, on j is (p_j - p_i)/d^2
        # We have diffs = p_i - p_j for i_idx, j_idx
        # Force on i: diffs / dists_sq
        # Force on j: -diffs / dists_sq
        
        np.add.at(grad_points, i_idx, diffs / dists_sq[:, np.newaxis])
        np.add.at(grad_points, j_idx, -diffs / dists_sq[:, np.newaxis])
        
        # Gradient of -sum_log_dists is -grad_points
        grad_val = -grad_points
        
        # Gradient of penalties
        # d/dp_k (sum ||p_i||^2) = 2 p_k
        grad_pen_scale = 20.0 * 2.0 * points # 10 * 2 * (norm_sq - target) * 2 * p ? 
        # Wait, pen_scale = (norm_sq - T)^2. d/dp = 2(norm_sq-T) * 2p = 4(norm_sq-T)p.
        # My penalty coeff was 10. So 10 * 4 * (norm_sq - T) * p = 40 * (norm_sq - T) * p
        grad_pen_scale = 40.0 * (norm_sq - target_norm_sq) * points
        
        # d/dp_k (sum ||mean(p)||^2) = d/dp_k (1/n^2 sum_i sum_j (p_i - p_k)(p_j - p_k) ...?)
        # Simpler: centroid = (1/n) sum p_i.
        # pen_centroid = ||c||^2.
        # d/dc = 2c. dc/dp_k = 1/n I.
        # d/dp_k = 2c * 1/n = 2/n c.
        # Penalty coeff 100.
        grad_pen_centroid = 100.0 * (2.0 / n) * centroid
        
        total_grad = grad_val + grad_pen_scale + grad_pen_centroid
        
        return val, total_grad.reshape(-1)

    # Restart loop
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        # Generate initial points
        # Random points in a square, centered
        np.random.seed(n_restarts)
        x0 = np.random.randn(n, d)
        x0 -= np.mean(x0, axis=0)
        # Scale to target norm
        norm = np.sqrt(np.sum(x0**2))
        if norm > 0:
            x0 = x0 * (np.sqrt(n) / norm)
        
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.status == 0:
            points = res.x.reshape(n, d)
            # Final cleanup: center and scale exactly
            points -= np.mean(points, axis=0)
            norm = np.sqrt(np.sum(points**2))
            if norm > 0:
                points = points * (np.sqrt(n) / norm)
            
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

        # If we have a very good score, we might stop early or just continue to find better
        # The theoretical max for 16 points is not 1, but close to 0.5-0.8 for R?
        # R = (dmin/dmax)^2.
        # For a regular hexagon packing, ratio is better.
        # If best_score > 0.5, it's a strong candidate.
        
        if time.time() > deadline - 10.0:
            break

    if best_points is None:
        # Fallback
        points = np.random.randn(n, d)
        points -= np.mean(points, axis=0)
        norm = np.sqrt(np.sum(points**2))
        points = points * (np.sqrt(n) / norm)
        return points

    return best_points

# EVOLVE-BLOCK-END
