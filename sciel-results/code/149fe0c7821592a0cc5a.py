# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    n = 16
    n_pairs = n * (n - 1) // 2

    # Precompute indices for pairwise distances to allow vectorized gradient calculation
    # i, j are indices into the flattened point array? No, indices into the points array.
    # Let's create arrays of point indices for all pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def score(pts):
        """Compute exact R = (d_min / d_max)^2"""
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_ratio_flat(flat_pts):
        """Nonsmooth objective for exact polishing"""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max) ** 2

    def smooth_objective_and_grad(flat_pts, beta):
        """
        Compute the smooth objective - (soft_min / soft_max)^2 and its analytic gradient.
        """
        pts = flat_pts.reshape(n, 2)
        
        # Compute pairwise vectors and distances
        # diff[i, j] = pts[i_idx[i]] - pts[j_idx[i]]
        diff = pts[i_idx] - pts[j_idx]
        d = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero in gradient if points coincide
        # If d is 0, gradient is undefined, but we can set it to 0 or small epsilon
        # For stability, replace 0 with 1e-12
        d_safe = np.where(d < 1e-12, 1e-12, d)
        
        # Soft min and max
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # soft_max = 1/beta * log(sum(exp(beta * d)))
        
        # Numerical stability for log-sum-exp
        # For min: max_val = max(-beta * d) = -beta * min(d)
        # For max: max_val = max(beta * d) = beta * max(d)
        
        neg_beta_d = -beta * d
        pos_beta_d = beta * d
        
        max_neg = np.max(neg_beta_d)
        max_pos = np.max(pos_beta_d)
        
        # Handle edge cases where max is -inf or inf (shouldn't happen with finite pts)
        if not np.isfinite(max_neg) or not np.isfinite(max_pos):
            # Fallback to exact score if numerical issues
            d_min = np.min(d)
            d_max = np.max(d)
            if d_max == 0: return 0.0, np.zeros_like(flat_pts)
            val = (d_min / d_max) ** 2
            return -val, np.zeros_like(flat_pts)

        sum_exp_neg = np.sum(np.exp(neg_beta_d - max_neg))
        sum_exp_pos = np.sum(np.exp(pos_beta_d - max_pos))
        
        soft_min = -(max_neg + np.log(sum_exp_neg)) / beta
        soft_max = (max_pos + np.log(sum_exp_pos)) / beta
        
        if soft_max <= 0:
            return 0.0, np.zeros_like(flat_pts)
            
        ratio = soft_min / soft_max
        obj = -(ratio ** 2)
        
        # Gradient calculation
        # d(soft_min)/dx = -beta * sum_i p_i^min * (diff_i / d_i)
        # d(soft_max)/dx =  beta * sum_i p_i^max * (diff_i / d_i)
        
        # p_i^min = exp(-beta * d_i) / sum(exp(-beta * d))
        # Note: exp(neg_beta_d - max_neg) / sum_exp_neg * exp(max_neg)
        # But we can just compute weights relative to the unnormalized sum?
        # No, we need the actual probabilities p_i.
        # p_i^min = exp(neg_beta_d) / sum_exp_neg_total
        # To avoid underflow/overflow, we can use the shifted values:
        # exp(neg_beta_d) = exp(neg_beta_d - max_neg) * exp(max_neg)
        # Sum = sum_exp_neg * exp(max_neg)
        # So p_i^min = exp(neg_beta_d - max_neg) / sum_exp_neg
        
        p_min = np.exp(neg_beta_d - max_neg) / sum_exp_neg
        p_max = np.exp(pos_beta_d - max_pos) / sum_exp_pos
        
        # Gradient of d_i with respect to points
        # d_i = || p_i - p_j ||
        # d(d_i)/d(p_k) = (p_k - p_j)/d_i if k=i, (p_i - p_k)/d_i if k=j, 0 otherwise.
        # We need grad of obj w.r.t flat_pts.
        
        # Let G_min be the vector sum of gradients of soft_min
        # G_min = -beta * sum_i p_min_i * (diff_i / d_i)
        # Note: diff_i = p_i - p_j.
        # grad w.r.t p_i is + (diff_i / d_i)
        # grad w.r.t p_j is - (diff_i / d_i)
        
        # We accumulate gradients into a (n, 2) array
        grad_pts_min = np.zeros((n, 2))
        grad_pts_max = np.zeros((n, 2))
        
        # diff_i / d_i is the unit vector from j to i
        # Handle d_i = 0
        unit_vecs = diff / d_safe[:, np.newaxis]
        
        # Contribution to p_i (index i_idx)
        # grad_pts_min[i_idx] += -beta * p_min * unit_vecs
        np.add.at(grad_pts_min, i_idx, -beta * p_min[:, np.newaxis] * unit_vecs)
        # Contribution to p_j (index j_idx)
        # grad_pts_min[j_idx] += -beta * p_min * (-unit_vecs) = +beta * p_min * unit_vecs
        np.add.at(grad_pts_min, j_idx, beta * p_min[:, np.newaxis] * unit_vecs)
        
        # For max:
        # G_max = beta * sum_i p_max_i * (diff_i / d_i)
        # grad_pts_max[i_idx] += beta * p_max * unit_vecs
        np.add.at(grad_pts_max, i_idx, beta * p_max[:, np.newaxis] * unit_vecs)
        # grad_pts_max[j_idx] += beta * p_max * (-unit_vecs)
        np.add.at(grad_pts_max, j_idx, -beta * p_max[:, np.newaxis] * unit_vecs)
        
        # Objective f = -(u/v)^2 where u=soft_min, v=soft_max
        # df/dx = -2 (u/v)^3 * (v * du/dx - u * dv/dx) / v^2
        # df/dx = -2 (u/v)^3 / v * (du/dx - (u/v) * dv/dx)
        # df/dx = -2 * ratio^3 / v * (grad_pts_min - ratio * grad_pts_max)
        
        factor = -2.0 * (ratio ** 3) / soft_max
        grad_pts = factor * (grad_pts_min - ratio * grad_pts_max)
        
        return obj, grad_pts.flatten()

    def make_hexagonal_initial():
        """
        Generate a 16-point configuration based on a hexagonal lattice.
        Center + 6 (r=1) + 9 (r=sqrt(3) approx).
        """
        pts = []
        pts.append([0.0, 0.0])
        
        for k in range(6):
            angle = k * 2 * np.pi / 6
            pts.append([np.cos(angle), np.sin(angle)])
            
        r_outer = np.sqrt(3)
        for k in range(9):
            angle = k * 2 * np.pi / 9
            pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
            
        pts = np.array(pts[:n])
        pts -= np.mean(pts, axis=0)
        return pts

    def make_grid_initial():
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        pts -= np.mean(pts, axis=0)
        return pts

    def make_circle_initial():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        pts -= np.mean(pts, axis=0)
        return pts

    def make_random_initial(seed):
        rng = np.random.RandomState(seed)
        pts = rng.rand(n, 2) * 2 - 1
        pts -= np.mean(pts, axis=0)
        return pts

    def improve_smooth(pts, betas):
        """
        Optimize using L-BFGS-B with analytic gradient, annealing beta.
        """
        current_pts = pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            # Define fun and jac for scipy
            def fun(x):
                val, _ = smooth_objective_and_grad(x, beta)
                return val
            
            def jac(x):
                _, g = smooth_objective_and_grad(x, beta)
                return g
            
            res = minimize(
                fun,
                current_pts.flatten(),
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            current_pts = res.x.reshape(n, 2)
            
        return current_pts

    def improve_exact(pts, max_iter=2000):
        """
        Final polish using Powell on the exact nonsmooth objective.
        """
        res = minimize(
            neg_ratio_flat,
            pts.flatten(),
            method='Powell',
            options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        return res.x.reshape(n, 2)

    # Generate initial configurations
    initial_configs = [
        make_hexagonal_initial(),
        make_grid_initial(),
        make_circle_initial(),
        make_random_initial(1),
        make_random_initial(42),
        make_random_initial(123),
        make_random_initial(999),
    ]
    
    # Add perturbed versions
    rng = np.random.RandomState(0)
    base_hex = make_hexagonal_initial()
    for noise_scale in [0.05, 0.1, 0.2]:
        perturbed = base_hex + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)
        
    base_grid = make_grid_initial()
    for noise_scale in [0.05, 0.1]:
        perturbed = base_grid + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    best_pts = None
    best_val = -np.inf

    # Betas for annealing: Start smooth, end sharp
    # Small beta = very smooth (convex-ish), Large beta = sharp (close to min/max)
    # We want to start with small beta to find the basin, then increase beta to sharpen.
    # Wait, standard annealing for this: Start with large beta (smooth) to get into basin, 
    # then decrease beta? No.
    # soft_min with beta->0 is mean. soft_min with beta->inf is min.
    # So we want to start with SMALL beta (smooth, mean-like) to find global structure?
    # Or start with LARGE beta (sharp) if we have a good init?
    # Actually, usually one starts with a very smooth approximation (small beta) to avoid local minima 
    # caused by the non-differentiability of min/max, then increases beta to sharpen.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # 1. Smooth optimization
        smooth_pts = improve_smooth(init_pts, betas)
        val = score(smooth_pts)
        
        if val > best_val:
            best_val = val
            best_pts = smooth_pts
            
        # 2. Exact polish on the result
        if time.time() < deadline - 15:
             polished_pts = improve_exact(smooth_pts, max_iter=500)
             polished_val = score(polished_pts)
             if polished_val > best_val:
                 best_val = polished_val
                 best_pts = polished_pts

    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 5:
        final_pts = improve_exact(best_pts, max_iter=1000)
        final_val = score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val
    
    if best_pts is None:
        best_pts = make_hexagonal_initial()
    
    return best_pts

# EVOLVE-BLOCK-END
