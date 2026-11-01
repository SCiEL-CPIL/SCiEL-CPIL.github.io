import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # 1. Initialization: Hexagonal lattice
    # Approximate spacing for N=300 in unit circle
    # r ~ 0.05, d ~ 0.1
    spacing = 0.1
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    
    if len(points) >= N:
        # Select N points closest to center
        points = points[np.argsort(radius, kind="stable")[:N]]
    else:
        # Fallback
        points = np.random.uniform(-1, 1, (N, 2))
        norms = np.linalg.norm(points, axis=1)
        points = points * (1.0 / np.maximum(norms, 1e-6))[:, np.newaxis] * 0.99

    x0 = points.flatten()
    
    def compute_v(x):
        pts = x.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d_vals = dist[iu] / 2.0
        norms = np.linalg.norm(pts, axis=1)
        b_vals = 1.0 - norms
        return np.concatenate([d_vals, b_vals])

    def objective_and_grad(x, beta):
        v = compute_v(x)
        u = -beta * v
        max_u = np.max(u)
        log_sum_exp = max_u + np.log(np.sum(np.exp(u - max_u)))
        f_val = log_sum_exp / beta
        
        log_w = u - log_sum_exp
        w = np.exp(log_w)
        
        pts = x.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-12] = 1e-12
        
        n_pairs = len(v) - N
        w_pair = w[:n_pairs]
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        W = np.zeros((N, N))
        W[i_idx, j_idx] = w_pair
        W[j_idx, i_idx] = w_pair
        
        U = diff / dist[:, :, np.newaxis]
        grad_pair = np.sum(W[:, :, np.newaxis] * U, axis=1)
        
        w_bound = w[n_pairs:]
        norms = np.linalg.norm(pts, axis=1)
        norms[norms < 1e-12] = 1e-12
        grad_bound = - (pts / norms[:, np.newaxis]) * w_bound[:, np.newaxis]
        
        grad_v = grad_pair + grad_bound
        grad_f = -grad_v
        
        return f_val, grad_f

    best_x = x0.copy()
    best_score = -np.inf
    
    # Adaptive beta schedule
    # Start low to find global structure, increase to refine
    betas = [5, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0], 
                best_x, 
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 10}
            )
            if res.fun < 1e10: # Check for explosion
                v = compute_v(res.x)
                score = np.min(v)
                if score > best_score:
                    best_score = score
                    best_x = res.x
        except Exception:
            continue
            
    pts = best_x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999999
        
    if not np.all(np.isfinite(pts)):
        # Fallback to initial if something went wrong
        return x0.reshape(N, 2)
        
    return pts

# EVOLVE-BLOCK-END
