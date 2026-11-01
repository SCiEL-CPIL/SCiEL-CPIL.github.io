import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Generate Base Configuration: Octahedron + Cube
    cube_verts = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    octa_verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    
    def rigid_score(s):
        pts = np.vstack([cube_verts, s * octa_verts])
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        return np.max(d2) / np.min(d2)

    # Find optimal s
    s_vals = np.linspace(0.1, 2.0, 200)
    scores = [rigid_score(s) for s in s_vals]
    best_s = s_vals[np.argmin(scores)]
    
    base_points = np.vstack([cube_verts, best_s * octa_verts])
    base_points -= np.mean(base_points, axis=0)
    
    best_score = np.inf
    best_points = base_points.copy()
    
    # Helper for optimization
    def objective_and_grad(x, tau):
        pts = x.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 0)
        
        mask = ~np.eye(N, dtype=bool)
        
        exp_plus = np.exp(tau * d2 * mask)
        exp_minus = np.exp(-tau * d2 * mask)
        
        sum_plus = np.sum(exp_plus)
        sum_minus = np.sum(exp_minus)
        
        val = (1.0 / tau) * (np.log(sum_plus) + np.log(sum_minus))
        
        w_plus = exp_plus / sum_plus
        w_minus = exp_minus / sum_minus
        
        np.fill_diagonal(w_plus, 0)
        np.fill_diagonal(w_minus, 0)
        
        sum_w_plus_x = w_plus @ pts
        sum_w_minus_x = w_minus @ pts
        
        sum_w_plus = np.sum(w_plus, axis=1)
        sum_w_minus = np.sum(w_minus, axis=1)
        
        grad = (1.0 / tau) * ( 2 * (pts * sum_w_plus[:, np.newaxis] - sum_w_plus_x) 
                             + 2 * (pts * sum_w_minus[:, np.newaxis] - sum_w_minus_x) )
        
        return val, grad

    def raw_score(x):
        pts = x.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d_min = np.min(d2)
        d_max = np.max(d2)
        return d_max / d_min

    # Multiple restarts
    np.random.seed(42)
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Add noise to base points
        noise = np.random.normal(0, 0.1, (N, D))
        points = base_points + noise
        points -= np.mean(points, axis=0)
        
        tau = 10.0
        max_tau = 1000.0
        
        while time.time() < deadline - 10.0:
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, tau)[0],
                    points,
                    jac=lambda x: objective_and_grad(x, tau)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                points = res.x
                
                current_score = raw_score(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = points.copy()
                    
            except Exception:
                pass
            
            tau *= 2.0
            if tau > max_tau:
                break
        
        # Final polish
        if time.time() < deadline - 5.0:
            res_nm = minimize(
                raw_score,
                best_points,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            final_score = raw_score(res_nm.x)
            if final_score < best_score:
                best_score = final_score
                best_points = res_nm.x.copy()

    return best_points
