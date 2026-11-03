import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    best_score = 0.0
    best_points = None
    
    def get_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist) / 2.0
        radii = np.sqrt(np.sum(points**2, axis=1))
        min_boundary = np.min(1.0 - radii)
        return min(min_pair_dist, min_boundary)

    def objective_neg_softmin(points, beta=100.0):
        p = points.reshape(N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 0.0)
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu] / 2.0
        radii = np.sqrt(np.sum(p**2, axis=1))
        boundary_clears = 1.0 - radii
        all_constraints = np.concatenate([pair_dists, boundary_clears])
        min_c = np.min(all_constraints)
        shifted = all_constraints - min_c
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        softmin_val = min_c - (1.0 / beta) * log_sum
        return -softmin_val

    def gradient_neg_softmin(points, beta=100.0):
        p = points.reshape(N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu] / 2.0
        radii = np.sqrt(np.sum(p**2, axis=1) + 1e-12)
        boundary_clears = 1.0 - radii
        all_constraints = np.concatenate([pair_dists, boundary_clears])
        min_c = np.min(all_constraints)
        shifted = all_constraints - min_c
        weights = np.exp(-beta * shifted)
        sum_weights = np.sum(weights)
        probs = weights / sum_weights
        grad_constraints = -probs
        grad_points = np.zeros_like(p)
        pair_probs = grad_constraints[:len(pair_dists)]
        boundary_probs = grad_constraints[len(pair_dists):]
        W = np.zeros((N, N))
        W[iu] = pair_probs
        W = W + W.T
        G = 0.5 * diff / dist[:, :, np.newaxis]
        np.fill_diagonal(G, 0.0)
        grad_points = np.sum(W[:, :, np.newaxis] * G, axis=1)
        grad_boundary = -p / radii[:, np.newaxis]
        grad_points += boundary_probs[:, np.newaxis] * grad_boundary
        return grad_points.reshape(-1)

    num_restarts = 30
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            k = int(1.0 / spacing) + 2
            rows = [((a + 0.5 * (b % 2)) * spacing, b * spacing * np.sqrt(3.0) / 2.0)
                    for a in range(-k, k + 1) for b in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) < N:
                 points = np.random.uniform(-0.8, 0.8, (N, 2))
            else:
                 idx = np.argsort(radius, kind="stable")[:N]
                 points = points[idx]
        else:
            angles = np.random.uniform(0, 2 * np.pi, N)
            rads = np.random.uniform(0, 0.9, N)
            points = np.column_stack([rads * np.cos(angles), rads * np.sin(angles)])
        
        res = minimize(
            objective_neg_softmin, 
            points.flatten(), 
            method='L-BFGS-B', 
            jac=gradient_neg_softmin,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        opt_points = res.x.reshape(N, 2)
        score = get_score(opt_points)
        
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
            
        if best_score > 0.125825:
             break

    if best_points is None:
         return np.zeros((N, 2))
         
    return best_points
