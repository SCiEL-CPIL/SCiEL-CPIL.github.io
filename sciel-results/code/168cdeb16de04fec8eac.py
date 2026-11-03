import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    # 1. Initialization: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98

    x0 = points.flatten()
    
    def get_points(x):
        return x.reshape(N, 2)

    def calc_score(x):
        p = get_points(x)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(p, axis=1))
        return min(min_pair, min_bound)

    def compute_softmin_and_grad(x, beta):
        points = get_points(x)
        
        # Compute pairwise distances efficiently
        # Use matrix multiplication: |a-b|^2 = |a|^2 + |b|^2 - 2*a.b
        sq_norms = np.sum(points * points, axis=1)  # (N,)
        dot_prod = points @ points.T  # (N, N)
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot_prod
        sq_dists = np.maximum(sq_dists, 0.0)
        dists = np.sqrt(sq_dists)
        
        iu = np.triu_indices(N, k=1)
        d_ij = dists[iu]
        
        # Boundary distances
        norms = np.sqrt(sq_norms)
        d_b = 1.0 - norms
        
        v = np.concatenate([d_ij / 2.0, d_b])
        
        # Softmin
        v_min = np.min(v)
        exp_v = np.exp(-beta * (v - v_min))
        S = np.sum(exp_v)
        weights = exp_v / S
        
        softmin_val = v_min - (1.0 / beta) * np.log(S)
        
        # Gradient
        w_pairs = weights[:len(d_ij)]
        w_bound = weights[len(d_ij):]
        
        # Pairwise gradient
        w_mat = np.zeros((N, N))
        w_mat[iu] = w_pairs
        w_sym = w_mat + w_mat.T
        
        d_safe = np.where(dists < 1e-9, 1e-9, dists)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        factor = w_sym / (2.0 * d_safe)
        np.fill_diagonal(factor, 0.0)
        
        grad_points = np.einsum('ij,ijk->ik', factor, diff)
        
        # Boundary gradient
        norms_safe = np.where(norms < 1e-9, 1e-9, norms)
        grad_bound_points = - (w_bound[:, np.newaxis] * points) / norms_safe[:, np.newaxis]
        
        total_grad = (grad_points + grad_bound_points).flatten()
        
        obj_val = -softmin_val
        grad_val = -total_grad
        
        return obj_val, grad_val

    best_x = x0.copy()
    best_score = calc_score(best_x)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Beta schedule with more iterations at low temperatures
    betas = [10, 50, 200, 1000, 5000, 10000, 20000]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Limit iterations based on remaining time
        remaining = deadline - time.time()
        max_iter = min(1000, int(remaining * 10))
        if max_iter < 50:
            break
        
        try:
            res = minimize(
                lambda x: compute_softmin_and_grad(x, beta)[0],
                current_x,
                jac=lambda x: compute_softmin_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'maxfun': max_iter * 2}
            )
            
            new_score = calc_score(res.x)
            if new_score > best_score:
                best_score = new_score
                best_x = res.x.copy()
                current_x = res.x.copy()
                
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Final polish: try a few more local refinements on the best solution
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    lambda x: compute_softmin_and_grad(x, 50000)[0],
                    best_x,
                    jac=lambda x: compute_softmin_and_grad(x, 50000)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400}
                )
                new_score = calc_score(res.x)
                if new_score > best_score:
                    best_score = new_score
                    best_x = res.x.copy()
            except Exception:
                pass
    
    return best_x.reshape(N, 2)
