import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, LinearConstraint

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 630s kill

    def score(pts):
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    # Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    
    initial_configs = []
    
    # Config 1: Icosahedron + 2 Poles (on unit sphere)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    cfg1 = np.vstack([ico12, poles])
    initial_configs.append(cfg1)
    
    # Config 2: Fibonacci sphere
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])
    
    fib14 = fibonacci_sphere(14)
    initial_configs.append(fib14)
    
    # Random restarts based on icosahedron
    rng = np.random.RandomState(42)
    for i in range(100):
        noise_level = 0.05 + 0.2 * (i / 100)
        pert_ico = ico12 + rng.randn(12, 3) * noise_level
        # Keep poles somewhat stable but allow noise
        pert_poles = poles + rng.randn(2, 3) * noise_level * 0.5
        cfg = np.vstack([pert_ico, pert_poles])
        initial_configs.append(cfg)
        
    # Pure random restarts
    for i in range(50):
        cfg = rng.randn(n, d)
        initial_configs.append(cfg)
    
    def objective(x, beta):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if np.max(dists) < 1e-12:
            return 0.0
        
        # Log-sum-exp trick for stability
        neg_d = -beta * dists
        max_neg_d = np.max(neg_d)
        log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
        soft_min = -log_sum_exp_neg / beta
        
        pos_d = beta * dists
        max_pos_d = np.max(pos_d)
        log_sum_exp_pos = max_pos_d + np.log(np.sum(np.exp(pos_d - max_pos_d)))
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return 0.0
        return -(soft_min / soft_max) ** 2
    
    def objective_grad(x, beta):
        pts = x.reshape((n, d))
        pts_centered = pts - np.mean(pts, axis=0)
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = pts_centered[i_idx] - pts_centered[j_idx]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        if np.max(dists) < 1e-12:
            return np.zeros_like(x)
        
        # Soft min/max weights
        neg_d = -beta * dists
        max_neg_d = np.max(neg_d)
        exp_neg = np.exp(neg_d - max_neg_d)
        sum_exp_neg = np.sum(exp_neg)
        w_min = exp_neg / sum_exp_neg
        
        pos_d = beta * dists
        max_pos_d = np.max(pos_d)
        exp_pos = np.exp(pos_d - max_pos_d)
        sum_exp_pos = np.sum(exp_pos)
        w_max = exp_pos / sum_exp_pos
        
        log_sum_exp_neg = max_neg_d + np.log(sum_exp_neg)
        soft_min = -log_sum_exp_neg / beta
        
        log_sum_exp_pos = max_pos_d + np.log(sum_exp_pos)
        soft_max = log_sum_exp_pos / beta
        
        if soft_max < 1e-12:
            return np.zeros_like(x)
        
        ratio = soft_min / soft_max
        
        # Gradient of ratio
        # c_k = (soft_max * w_min_k - soft_min * w_max_k) / soft_max^2
        c = (soft_max * w_min - soft_min * w_max) / (soft_max ** 2)
        
        # Vectorized scatter
        unit_diffs = diffs / dists[:, None]
        contrib = c[:, None] * unit_diffs
        
        grad = np.zeros((n, d))
        np.add.at(grad, i_idx, contrib)
        np.add.at(grad, j_idx, -contrib)
        
        # Centering constraint gradient
        sum_grad = np.sum(grad, axis=0)
        grad = grad - (1.0 / n) * sum_grad
        
        # Objective is -ratio^2
        final_grad = -2.0 * ratio * grad
        
        return final_grad.flatten()
    
    best_points = None
    best_score = -np.inf
    
    # Optimize from multiple initial configurations
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break
        
        current_pts = init.copy()
        
        # Stage 1: Coarse optimization with increasing beta
        # Start with low beta for global landscape, increase for sharpness
        betas = [10, 50, 200, 500, 1000, 2000, 5000, 10000]
        for beta in betas:
            if time.time() > deadline - 30:
                break
                
            try:
                result = minimize(
                    lambda x, b=beta: objective(x, b),
                    current_pts.flatten(),
                    jac=lambda x, b=beta: objective_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
                )
                
                if np.all(np.isfinite(result.x)):
                    new_pts = result.x.reshape((n, d))
                    if np.all(np.isfinite(new_pts)):
                        current_pts = new_pts
                        
            except Exception:
                pass
        
        # Evaluate and update best
        s = score(current_pts)
        if s > best_score:
            best_score = s
            best_points = current_pts.copy()
            
        # Stage 2: Fine optimization with Nelder-Mead on exact objective
        if time.time() < deadline - 15:
            def obj_nm(x):
                pts_nm = x.reshape((n, d))
                return -score(pts_nm)
            
            try:
                result_nm = minimize(
                    obj_nm,
                    current_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                pts_nm = result_nm.x.reshape((n, d))
                if np.all(np.isfinite(pts_nm)):
                    s_nm = score(pts_nm)
                    if s_nm > best_score:
                        best_score = s_nm
                        best_points = pts_nm.copy()
                        
            except Exception:
                pass
        
        # Stage 3: Local Search / Perturbation
        if time.time() < deadline - 5 and best_points is not None:
            # Try a few random perturbations of the current best local solution
            # This helps escape local minima of the smoothed objective
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                noise = rng.randn(n, d) * 0.01
                cand_pts = best_points + noise
                s_cand = score(cand_pts)
                if s_cand > best_score:
                    best_score = s_cand
                    best_points = cand_pts.copy()

    if best_points is None:
        best_points = initial_configs[0]
        
    # Final normalization: center only
    best_points = best_points - np.mean(best_points, axis=0)

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
