# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s buffer
    
    n = 14
    d = 3
    
    def compute_metric(points):
        """
        Computes the squared ratio (d_min / d_max)^2.
        Returns 0 if invalid.
        """
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or d_min == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_and_grad(points_flat, k):
        """
        Smoothed objective function and its gradient.
        Minimizes approx log(d_max) - log(d_min).
        """
        points = points_flat.reshape(n, d)
        
        # Center points to remove translation invariance
        # Note: The objective is translation invariant, but gradient methods might drift.
        # We can project out the center of mass in the gradient or just let it drift and re-center at the end.
        # For now, we assume the optimizer handles it or we add a small penalty for non-zero center.
        
        dists = pdist(points)
        
        # Soft max/min approximations
        # To avoid overflow, we shift by max/min
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # log sum exp(k * d)
        # exp(k * (d - max_d)) * exp(k * max_d)
        # log(sum(exp(k*d))) = k*max_d + log(sum(exp(k*(d-max_d))))
        term_plus = k * max_d + np.log(np.sum(np.exp(k * (dists - max_d))))
        
        # log sum exp(-k * d)
        # exp(-k * (d - min_d)) * exp(-k * min_d)
        # log(sum(exp(-k*d))) = -k*min_d + log(sum(exp(-k*(d-min_d))))
        term_minus = -k * min_d + np.log(np.sum(np.exp(-k * (dists - min_d))))
        
        # Objective: (1/k) * term_plus + (1/k) * term_minus
        # This approximates log(d_max) - log(d_min)
        val = (term_plus + term_minus) / k
        
        # Gradient calculation is complex with pdist.
        # We can use finite differences or derive it.
        # Given time constraints, let's use a numerical gradient or a simpler optimizer that doesn't need gradient.
        # Actually, SLSQP or Nelder-Mead might be safer if gradient is tricky.
        # But L-BFGS-B is faster. Let's try to compute gradient.
        
        # Gradient of sum of exp(k * d_ij) w.r.t points
        # d/dp_i exp(k * d_ij) = exp(k * d_ij) * k * (p_i - p_j) / d_ij
        
        # This is O(N^2 * D) per point, total O(N^3 * D) which is fine for N=14.
        
        grad = np.zeros_like(points)
        
        # Gradient of term_plus / k
        # d/dp_i (1/k) log(sum exp(k d)) = (1/k) * (1/sum) * sum_j exp(k d_ij) * k * (p_i - p_j)/d_ij
        # = (1/sum) * sum_j exp(k d_ij) * (p_i - p_j)/d_ij
        
        # Gradient of term_minus / k
        # d/dp_i (1/k) log(sum exp(-k d)) = (1/k) * (1/sum) * sum_j exp(-k d_ij) * (-k) * (p_i - p_j)/d_ij
        # = -(1/sum) * sum_j exp(-k d_ij) * (p_i - p_j)/d_ij
        
        # We need to be careful with indices. pdist returns distances for i<j.
        # It's easier to construct the full distance matrix.
        
        # Reconstruct full distance matrix
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Weights for plus term
        exp_plus = np.exp(k * (dists_mat - max_d))
        exp_plus[~mask] = 0
        sum_plus = np.sum(exp_plus)
        w_plus = exp_plus / sum_plus
        
        # Weights for minus term
        exp_minus = np.exp(-k * (dists_mat - min_d))
        exp_minus[~mask] = 0
        sum_minus = np.sum(exp_minus)
        w_minus = exp_minus / sum_minus
        
        # Gradient contribution from each pair (i,j)
        # Term: w[i,j] * (p_i - p_j) / d_ij
        # Note: w[i,j] = w[j,i]
        # Contribution to grad[i] from j: w[i,j] * (p_i - p_j) / d_ij
        # Contribution to grad[j] from i: w[i,j] * (p_j - p_i) / d_ij = - w[i,j] * (p_i - p_j) / d_ij
        
        # Vectorized:
        # G_plus[i] = sum_j w_plus[i,j] * (p_i - p_j) / d_ij
        # G_minus[i] = - sum_j w_minus[i,j] * (p_i - p_j) / d_ij
        
        # Avoid division by zero
        dists_mat_safe = np.where(dists_mat > 1e-12, dists_mat, 1.0)
        
        # Factor (p_i - p_j) / d_ij
        # diff[i,j] / dists_mat[i,j]
        # This is the unit vector from j to i
        
        # We need to sum w[i,j] * unit_vec[i,j] for all j
        # unit_vec = diff / dists_mat_safe[:, :, np.newaxis]
        unit_vec = diff / dists_mat_safe[:, :, np.newaxis]
        
        # Mask out diagonal
        unit_vec[~mask, :] = 0
        
        # grad_plus[i] = sum_j w_plus[i,j] * unit_vec[i,j]
        grad_plus = np.sum(w_plus[:, :, np.newaxis] * unit_vec, axis=1)
        
        # grad_minus[i] = - sum_j w_minus[i,j] * unit_vec[i,j]
        grad_minus = -np.sum(w_minus[:, :, np.newaxis] * unit_vec, axis=1)
        
        total_grad = grad_plus + grad_minus
        
        return val, total_grad.flatten()

    best_score = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random on sphere
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= norms
        inits.append(pts)
        
    # 2. Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (i + 0.5) * 2 / n
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    inits.append(fibonacci_sphere(n))
    
    # 3. Lattice subset
    # Create a grid and pick 14 points
    grid_size = 3
    grid = np.array(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))).T.reshape(-1, 3)
    # Shuffle and take first 14
    np.random.shuffle(grid)
    inits.append(grid[:n])
    
    # Run optimization for each init
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Start with small k, increase it
        current_k = 10.0
        current_pts = init_pts.copy()
        
        # Optimization loop
        for step in range(20):
            if time.time() > deadline:
                break
                
            # Define the objective for this k
            def obj_func(x):
                val, _ = objective_and_grad(x, current_k)
                return val
            
            def grad_func(x):
                _, g = objective_and_grad(x, current_k)
                return g
                
            try:
                res = minimize(obj_func, current_pts.flatten(), jac=grad_func, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success:
                    current_pts = res.x.reshape(n, d)
            except Exception:
                pass
            
            # Increase k to sharpen
            current_k *= 2.0
            if current_k > 1000:
                current_k = 1000
                
        # Evaluate the actual metric
        score = compute_metric(current_pts)
        if score > best_score:
            best_score = score
            best_points = current_pts.copy()
            
    # If no points found or score is 0, return random
    if best_points is None:
        best_points = np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
