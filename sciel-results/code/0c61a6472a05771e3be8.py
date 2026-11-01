import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialize with Hexagonal Lattice
    def generate_hex_grid(N_target, r_init):
        spacing = 2.0 * r_init
        R_est = spacing * np.sqrt(N_target * np.sqrt(3.0) / (2.0 * np.pi))
        R_est *= 1.1  # Add margin
        
        k = int(np.ceil(R_est / spacing)) + 2
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > R_est:
                continue
            x_offset = (j % 2) * spacing / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if np.hypot(x, y) <= R_est:
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) == 0:
            return np.zeros((N_target, 2))
            
        # If we have more than N, select N closest to center
        if len(points) > N_target:
            radii = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii)
            points = points[indices[:N_target]]
        elif len(points) < N_target:
            while len(points) < N_target:
                angle = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, 1)
                points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
        
        return points

    # Initial guess for radius
    r_guess = 0.05
    points = generate_hex_grid(N, r_guess)
    
    # Ensure points are strictly inside
    max_rad = np.max(np.hypot(points[:, 0], points[:, 1]))
    if max_rad > 1.0:
        scale = 0.99 / max_rad
        points *= scale

    best_points = points.copy()
    best_score = -np.inf

    # Define objective function factory
    def make_objective(beta):
        def objective(x):
            p = x.reshape(-1, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
            
            np.fill_diagonal(dists, np.inf)
            min_dists = dists / 2.0
            valid_dists = min_dists[np.triu_indices(N, 1)]
            
            all_vals = np.concatenate([valid_dists, boundary_dists])
            m = np.min(all_vals)
            if m < 0:
                return m
            exp_vals = np.exp(-beta * (all_vals - m))
            soft_min = m - (1.0/beta) * np.log(np.sum(exp_vals))
            return -soft_min
        return objective

    def make_grad(beta):
        def objective_grad(x):
            p = x.reshape(-1, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1.0)
            boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
            
            valid_dists = dists / 2.0
            all_vals = np.concatenate([valid_dists, boundary_dists])
            m = np.min(all_vals)
            exp_vals = np.exp(-beta * (all_vals - m))
            S = np.sum(exp_vals)
            weights = exp_vals / S
            
            n_pairs = len(valid_dists)
            w_dists = weights[:n_pairs]
            w_bounds = weights[n_pairs:]
            
            grad = np.zeros_like(x)
            
            norms = np.linalg.norm(p, axis=1)
            norms[norms == 0] = 1.0
            grad_bounds = -p / norms[:, np.newaxis] * w_bounds[:, np.newaxis]
            grad += grad_bounds
            
            W_full = np.zeros((N, N))
            i_idx, j_idx = np.triu_indices(N, 1)
            W_full[i_idx, j_idx] = w_dists
            W_full[j_idx, i_idx] = w_dists
            
            inv_dists = 1.0 / (2.0 * dists)
            np.fill_diagonal(inv_dists, 0.0)
            
            coeff = W_full * inv_dists
            grad_dists = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            grad += grad_dists
            return grad
        return objective_grad

    x0 = points.flatten()
    bounds = [(-1.1, 1.1)] * (2 * N)
    
    # Beta schedule: Start smooth, end sharp
    beta_schedule = [10.0, 50.0, 100.0, 200.0, 400.0]
    
    current_x = x0
    
    for beta in beta_schedule:
        if time.time() > deadline - 10: # Reserve time for final checks
            break
            
        obj_func = make_objective(beta)
        grad_func = make_grad(beta)
        
        try:
            res = minimize(obj_func, current_x, jac=grad_func, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            if res.success or res.fun < -best_score:
                current_x = res.x
                best_points = current_x.reshape(-1, 2)
                # Calculate actual score for tracking
                p = best_points
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                min_dists = dists / 2.0
                valid_dists = min_dists[np.triu_indices(N, 1)]
                boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
                score = np.min(np.concatenate([valid_dists, boundary_dists]))
                best_score = score
        except Exception as e:
            pass

    # Final check: ensure points are valid (inside circle)
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii > 1.0):
        max_r = np.max(radii)
        if max_r > 1.0:
            scale = 0.999 / max_r
            best_points *= scale

    return best_points
