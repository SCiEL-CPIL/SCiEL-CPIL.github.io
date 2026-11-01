import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Step 1: Generate hexagonal lattice points
    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n]
            return points[idx].copy()
        return np.random.rand(n, 2) * 2 - 1

    pts = hex_lattice_init(N)
    best_pts = pts.copy()
    
    def compute_radius(pts):
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)
    
    best_radius = compute_radius(best_pts)
    current_pts = pts.copy()

    # Soft-min optimization
    def objective_and_grad(x_flat, beta):
        pts = x_flat.reshape(N, 2)
        
        # 1. Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, 1)
        pair_dists = dist_mat[iu]
        
        # 2. Compute boundary distances
        radii = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - radii
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Compute weights
        max_d = np.max(all_dists)
        exps = np.exp(-beta * (all_dists - max_d))
        sum_exps = np.sum(exps)
        weights = exps / sum_exps
        
        soft_min_val = max_d - np.log(sum_exps) / beta
        obj = -soft_min_val
        
        # Gradient
        grad = np.zeros_like(pts)
        
        weights_pair = weights[:len(pair_dists)]
        weights_bound = weights[len(pair_dists):]
        
        # Pairwise gradient
        W_mat = np.zeros((N, N))
        W_mat[iu] = weights_pair
        W_mat = W_mat + W_mat.T
        
        dist_mat_safe = dist_mat.copy()
        dist_mat_safe[np.diag_indices(N)] = 1.0
        
        inv_dist = W_mat / dist_mat_safe
        sum_w = np.sum(inv_dist, axis=1)
        weighted_pos = inv_dist @ pts
        
        grad_pair = pts * sum_w[:, np.newaxis] - weighted_pos
        
        # Boundary gradient
        radii_safe = radii.copy()
        radii_safe[radii_safe == 0] = 1.0
        unit_vec = pts / radii_safe[:, np.newaxis]
        grad_bound = -weights_bound[:, np.newaxis] * unit_vec
        
        grad_total = grad_pair + grad_bound
        
        return obj, grad_total

    beta = 20.0
    no_improve_count = 0
    
    while time.time() < deadline:
        def obj_func(x):
            val, _ = objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, g = objective_and_grad(x, beta)
            return g
        
        res = minimize(obj_func, current_pts.flatten(), method='L-BFGS-B', 
                       jac=grad_func, options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
        
        if res.success or res.status == 0:
            new_pts = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_pts)):
                r_new = compute_radius(new_pts)
                if r_new > best_radius + 1e-10:
                    best_radius = r_new
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Increase beta
        beta *= 1.3
        if beta > 100:
            beta = 100
            
        # If stuck, try a small perturbation
        if no_improve_count > 5:
            current_pts += np.random.normal(0, 0.001, size=(N, 2))
            no_improve_count = 0

    return best_pts
