# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin of 20 seconds

    n = 16

    def compute_score(points):
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_and_grad(x, beta):
        """
        Smooth surrogate: maximize soft_min / soft_max using log-sum-exp.
        Returns (objective_value, gradient) where we minimize negative of the ratio.
        """
        points = x.reshape(n, 2)
        d = pdist(points)
        
        # Avoid zero distances
        d_safe = np.maximum(d, 1e-12)
        
        # Soft min via log-sum-exp: soft_min = -log(sum(exp(-beta*d)))/beta
        # For numerical stability
        d_min_shift = d_safe - np.min(d_safe)
        log_sum_exp_min = np.log(np.sum(np.exp(-beta * d_min_shift))) + np.min(d_safe)
        soft_min = log_sum_exp_min / beta
        
        # Soft max via log-sum-exp: soft_max = log(sum(exp(beta*d)))/beta
        d_max_shift = d_safe - np.max(d_safe)
        log_sum_exp_max = np.log(np.sum(np.exp(beta * d_max_shift))) + np.max(d_safe)
        soft_max = log_sum_exp_max / beta
        
        if soft_max < 1e-12:
            return 0.0, np.zeros_like(x)
        
        ratio = soft_min / soft_max
        score = ratio * ratio
        
        # Gradient computation
        # d(soft_min)/dx = -1/beta * sum_k w_k_min * d(d_k)/dx, where w_k_min = exp(-beta*d_k)/sum(exp(-beta*d_j))
        # d(soft_max)/dx = 1/beta * sum_k w_k_max * d(d_k)/dx, where w_k_max = exp(beta*d_k)/sum(exp(beta*d_j))
        
        # Weights for soft min
        exp_min = np.exp(-beta * d_min_shift)
        w_min = exp_min / np.sum(exp_min)
        
        # Weights for soft max
        exp_max = np.exp(beta * d_max_shift)
        w_max = exp_max / np.sum(exp_max)
        
        # Build gradients
        grad_soft_min = np.zeros_like(x)
        grad_soft_max = np.zeros_like(x)
        idx = np.triu_indices(n, k=1)
        
        for k in range(len(idx[0])):
            i, j = idx[0][k], idx[1][k]
            diff = points[i] - points[j]
            dist = d_safe[k]
            if dist < 1e-12:
                continue
            
            # Gradient of d_k w.r.t point i: diff/dist, w.r.t point j: -diff/dist
            grad_i_min = (w_min[k] / beta) * (diff / dist)
            grad_j_min = -(w_min[k] / beta) * (diff / dist)
            
            grad_i_max = (w_max[k] / beta) * (diff / dist)
            grad_j_max = -(w_max[k] / beta) * (diff / dist)
            
            grad_soft_min[2*i:2*i+2] += grad_i_min
            grad_soft_min[2*j:2*j+2] += grad_j_min
            
            grad_soft_max[2*i:2*i+2] += grad_i_min
            grad_soft_max[2*j:2*j+2] += grad_j_min
        
        # d(score)/dx = 2*ratio * (d(soft_min)/dx * soft_max - soft_min * d(soft_max)/dx) / soft_max^2
        # = 2*soft_min/soft_max * (grad_soft_min * soft_max - soft_min * grad_soft_max) / soft_max^2
        # = 2*soft_min/soft_max^2 * (grad_soft_min * soft_max - soft_min * grad_soft_max) / soft_max
        # Simplifying: d(score)/dx = 2 * ratio / soft_max * (grad_soft_min - ratio * grad_soft_max)
        
        grad_score = 2 * ratio / soft_max * (grad_soft_min - ratio * grad_soft_max)
        
        # We minimize -score
        return -score, -grad_score

    def optimize_from(points_init, beta_schedule=None, max_iters=150):
        x0 = points_init.flatten()
        
        if beta_schedule is None:
            beta_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
        
        best_x = x0.copy()
        best_score = compute_score(points_init)
        
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: objective_and_grad(x, b)[0],
                    x0,
                    jac=lambda x, b=beta: objective_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': max_iters, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                x0 = res.x
                points = x0.reshape(n, 2)
                score = compute_score(points)
                
                if score > best_score:
                    best_score = score
                    best_x = x0.copy()
            except Exception:
                continue
        
        return best_x.reshape(n, 2), best_score

    def regular_polygon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def concentric_rings(r_inn, n_outer, n_inner):
        """Create concentric ring configuration."""
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = r_inn * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def two_circles(r1, r2, n1, n2, phase=0):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phase
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def random_points(seed):
        np.random.seed(seed)
        return np.random.randn(n, 2)

    # Generate diverse initializations
    initializations = []
    
    # Regular polygon
    initializations.append(regular_polygon())
    
    # Concentric rings with various inner radii
    for r_inn in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        initializations.append(concentric_rings(r_inn, 11, 5))
    
    # Two circles with various radii and phases
    for r1, r2, n1, n2, phase in [
        (1.0, 0.5, 8, 8, 0),
        (1.0, 0.5, 8, 8, np.pi/8),
        (1.0, 0.6, 10, 6, 0),
        (1.0, 0.6, 10, 6, np.pi/10),
        (1.0, 0.4, 12, 4, 0),
        (1.0, 0.7, 6, 10, np.pi/6),
    ]:
        initializations.append(two_circles(r1, r2, n1, n2, phase))
    
    # Random initializations
    for seed in range(15):
        initializations.append(random_points(seed))
    
    # Grid-based
    for seed in range(5):
        np.random.seed(seed)
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < n:
                    pts.append([i, j])
        pts = np.array(pts[:n], dtype=float)
        pts += np.random.randn(n, 2) * 0.3
        initializations.append(pts)
    
    best_points = None
    best_score = -1.0
    
    for i, init in enumerate(initializations):
        if time.time() > deadline:
            break
        
        points, score = optimize_from(init)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        return regular_polygon()
    
    # Final polish with higher precision
    if time.time() < deadline - 30:
        points, score = optimize_from(best_points, beta_schedule=[1000.0, 5000.0, 10000.0], max_iters=200)
        if score > best_score:
            best_points = points
    
    return best_points


# EVOLVE-BLOCK-END
