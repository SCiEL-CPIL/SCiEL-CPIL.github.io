# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin

    n = 16
    d = 2

    # Use a smooth surrogate: maximize log(d_min) - log(d_max)
    # or equivalently minimize -log(d_min) + log(d_max)
    # But d_min and d_max are nonsmooth. Use soft-min and soft-max.
    
    # For a set of distances d_i, soft_min = -logsumexp(-d_i/beta) * beta
    # soft_max = logsumexp(d_i/beta) * beta
    # We want to maximize soft_min / soft_max, or maximize log(soft_min) - log(soft_max)
    
    # Actually, let's use a different approach:
    # Maximize t such that all d_i >= t * d_max
    # Equivalently, minimize -log(t) subject to d_i >= t * d_max for all i
    # But this is still nonsmooth.
    
    # Better: use the objective R = (d_min/d_max)^2
    # Maximize log(R) = 2*(log(d_min) - log(d_max))
    # Use soft approximations for d_min and d_max
    
    def compute_soft_min_max(points, beta):
        """Compute soft min and max of pairwise distances."""
        dists = pdist(points)
        # Soft min: -beta * log(sum(exp(-d/beta)))
        # Soft max: beta * log(sum(exp(d/beta)))
        # Use log-sum-exp trick for numerical stability
        neg_d = -dists / beta
        neg_d_max = np.max(neg_d)
        soft_min = -beta * (neg_d_max + np.log(np.sum(np.exp(neg_d - neg_d_max))))
        
        d_scaled = dists / beta
        d_max_scaled = np.max(d_scaled)
        soft_max = beta * (d_max_scaled + np.log(np.sum(np.exp(d_scaled - d_max_scaled))))
        
        return soft_min, soft_max
    
    def objective_and_grad(points_flat, beta):
        """Objective: maximize log(soft_min) - log(soft_max)
        Returns negative for minimization."""
        points = points_flat.reshape(-1, 2)
        soft_min, soft_max = compute_soft_min_max(points, beta)
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10, np.zeros_like(points_flat)
        
        val = -(np.log(soft_min) - np.log(soft_max))
        
        # Gradient via finite differences would be expensive; use analytic or skip
        # For now, return just the value and let L-BFGS-B use numerical gradient
        return val, None
    
    # Initialize with points on a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = init_points.copy()
    best_score = -1e10
    
    def score_points(points):
        """Compute the actual score R = (d_min/d_max)^2."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    best_score = score_points(best_points)
    
    # Try multiple beta values and initializations
    betas = [0.01, 0.05, 0.1, 0.2, 0.5]
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        # Different initializations
        if restart == 0:
            # Circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            init = np.column_stack([np.cos(angles), np.sin(angles)])
        elif restart == 1:
            # Random
            np.random.seed(restart * 100 + 1)
            init = np.random.randn(n, 2)
            init /= np.max(np.abs(init))
        elif restart == 2:
            # Two concentric circles
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            init1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
            init2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
            init = np.vstack([init1, init2])
        elif restart == 3:
            # Grid-like
            init = np.array([[i % 4, i // 4] for i in range(16)], dtype=float)
            init = init - np.mean(init, axis=0)
            init /= np.max(np.abs(init))
        else:
            # Random with seed
            np.random.seed(restart * 100 + 42)
            init = np.random.randn(n, 2)
            init /= np.max(np.abs(init))
        
        for beta in betas:
            if time.time() > deadline - 60:
                break
            
            x0 = init.flatten()
            
            # Use L-BFGS-B with numerical gradient
            try:
                result = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    opt_points = result.x.reshape(n, 2)
                    s = score_points(opt_points)
                    if s > best_score:
                        best_score = s
                        best_points = opt_points.copy()
                        
            except Exception:
                pass
            
            # Also try a simple gradient ascent with the smooth objective
            if time.time() > deadline - 60:
                break
            
            # Custom optimization: gradient ascent on log(soft_min) - log(soft_max)
            points = init.copy()
            lr = 0.01
            for it in range(2000):
                if time.time() > deadline - 60:
                    break
                
                dists = pdist(points)
                if len(dists) == 0:
                    break
                
                # Soft min and max
                neg_d = -dists / beta
                neg_d_max = np.max(neg_d)
                exp_neg = np.exp(neg_d - neg_d_max)
                sum_exp_neg = np.sum(exp_neg)
                
                d_scaled = dists / beta
                d_max_scaled = np.max(d_scaled)
                exp_pos = np.exp(d_scaled - d_max_scaled)
                sum_exp_pos = np.sum(exp_pos)
                
                # Gradient of log(soft_min) w.r.t. points
                # soft_min = -beta * (neg_d_max + log(sum_exp_neg))
                # d(soft_min)/d(points) = beta * d(log(sum_exp_neg))/d(points)
                # = beta * sum(exp_neg_i * d(neg_d_i)/d(points)) / sum_exp_neg
                # neg_d_i = -d_i / beta, so d(neg_d_i)/d(points) = -d(d_i)/d(points) / beta
                # So d(soft_min)/d(points) = -sum(exp_neg_i * d(d_i)/d(points)) / sum_exp_neg
                
                # Similarly for soft_max
                # soft_max = beta * (d_max_scaled + log(sum_exp_pos))
                # d(soft_max)/d(points) = beta * sum(exp_pos_i * d(d_i/d_beta)/d(points)) / sum_exp_pos
                # = sum(exp_pos_i * d(d_i)/d(points)) / sum_exp_pos
                
                # We need d(d_i)/d(points) for each pair
                # This is expensive; let's just do a simple heuristic update
                
                # Find nearest neighbor for each point
                np.fill_diagonal(np.eye(n), 0)
                dist_matrix = np.zeros((n, n))
                idx = np.triu_indices(n, k=1)
                dist_matrix[idx] = dists
                dist_matrix[idx[1], idx[0]] = dists
                
                # Move points away from their nearest neighbors and toward farther ones
                for i in range(n):
                    nearest = np.argmin(np.where(np.eye(n, dtype=bool), np.inf, dist_matrix[i]))
                    if i == nearest:
                        continue
                    # Move away from nearest
                    diff = points[i] - points[nearest]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        points[i] += lr * diff / norm
                
                # Normalize to keep scale
                max_norm = np.max(np.abs(points))
                if max_norm > 0:
                    points /= max_norm
                
                s = score_points(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
            
            # Check time
            if time.time() > deadline - 30:
                break
    
    # Final polish: try to improve with local search
    if time.time() < deadline - 10:
        # Try small perturbations
        best = best_points.copy()
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            np.random.seed(np.random.randint(10000))
            perturbed = best + 0.01 * np.random.randn(n, 2)
            s = score_points(perturbed)
            if s > best_score:
                best_score = s
                best = perturbed.copy()
    
    return best

# EVOLVE-BLOCK-END
