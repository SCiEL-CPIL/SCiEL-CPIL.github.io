import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 110.0  # Keep a 10s safety margin

    # Initial grid: 5x5x4
    x_vals = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_vals = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    z_vals = np.array([-0.375, -0.125, 0.125, 0.375])
    
    points = np.array([(x, y, z) for z in z_vals for y in y_vals for x in x_vals])
    # points shape (100, 3)
    
    def objective(params):
        c = params[:-1].reshape(-1, 3)
        r = params[-1]
        
        # Calculate pairwise distances using upper triangle
        i, j = np.triu_indices(N, k=1)
        diff = c[i] - c[j]
        dists = np.linalg.norm(diff, axis=1)
        
        # Soft-min penalty for pairs: log(1 + exp(-lambda * (dist - 2r)))
        lam = 50.0
        x_pairs = -lam * (dists - 2.0 * r)
        pair_terms = np.log1p(np.exp(np.clip(x_pairs, -500, 500)))
        
        # Boundary penalties: log(1 + exp(-lambda * (0.5 - |c_ik| - r)))
        abs_c = np.abs(c)
        margins = 0.5 - abs_c - r
        x_bound = -lam * margins
        bound_terms = np.log1p(np.exp(np.clip(x_bound, -500, 500)))
        
        sum_pairs = np.sum(pair_terms)
        sum_bound = np.sum(bound_terms)
        
        # Maximize r by subtracting alpha * r
        alpha = 1.0
        return sum_pairs + sum_bound - alpha * r

    def gradient(params):
        c = params[:-1].reshape(-1, 3)
        r = params[-1]
        
        i, j = np.triu_indices(N, k=1)
        diff = c[i] - c[j]
        dists = np.linalg.norm(diff, axis=1)
        
        lam = 50.0
        x_pairs = -lam * (dists - 2.0 * r)
        
        # Sigmoid function
        sig_pairs = 1.0 / (1.0 + np.exp(-np.clip(x_pairs, -500, 500)))
        
        # Gradient w.r.t c for pair terms
        grad_c = np.zeros_like(c)
        coeff = -lam * sig_pairs / dists
        np.add.at(grad_c, i, coeff[:, np.newaxis] * diff)
        np.add.at(grad_c, j, -coeff[:, np.newaxis] * diff)
        
        # Gradient w.r.t r for pair terms
        grad_r = np.sum(sig_pairs * 2.0 * lam)
        
        # Boundary gradient
        abs_c = np.abs(c)
        margins = 0.5 - abs_c - r
        x_bound = -lam * margins
        sig_bound = 1.0 / (1.0 + np.exp(-np.clip(x_bound, -500, 500)))
        
        # d_x_bound/d_c_ik = lam * sign(c_ik)
        sign_c = np.sign(c)
        grad_c += sig_bound * lam * sign_c
        
        # d_x_bound/d_r = lam
        grad_r += np.sum(sig_bound * lam)
        
        # Subtract alpha for maximizing r
        alpha = 1.0
        grad_r -= alpha
        
        grad = np.concatenate([grad_c.flatten(), [grad_r]])
        return grad

    def compute_actual_radius(c):
        """Compute the actual valid radius for a configuration."""
        i, j = np.triu_indices(N, k=1)
        dists = np.linalg.norm(c[i] - c[j], axis=1)
        min_dist = np.min(dists)
        min_bound = np.min(0.5 - np.abs(c))
        return min(min_dist / 2.0, min_bound)

    # Initial parameters
    r0 = 0.1
    x0 = np.concatenate([points.flatten(), [r0]])
    
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]
    
    best_r = compute_actual_radius(points)
    best_c = points.copy()
    
    # Run main optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
        
        if np.all(np.isfinite(res.x)):
            c_opt = res.x[:-1].reshape(-1, 3)
            actual_r = compute_actual_radius(c_opt)
            
            if actual_r > best_r:
                best_r = actual_r
                best_c = c_opt
    except Exception:
        pass

    # Random restarts with different initial configurations
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        rng = np.random.RandomState(42)
        
        for restart in range(10):
            if time.time() - start_time > deadline - 15:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Perturbed grid
                c_init = points + rng.normal(0, 0.03, points.shape)
            elif restart == 1:
                # FCC-like lattice initialization
                # FCC: face-centered cubic structure
                a = 0.25  # lattice constant
                fcc_points = []
                for ix in range(4):
                    for iy in range(4):
                        for iz in range(4):
                            # Corner positions
                            fcc_points.append([ix*a - 0.5 + a/2, iy*a - 0.5 + a/2, iz*a - 0.5 + a/2])
                            # Face centers
                            if ix < 3:
                                fcc_points.append([ix*a - 0.5 + a, iy*a - 0.5 + a/2, iz*a - 0.5 + a/2])
                            if iy < 3:
                                fcc_points.append([ix*a - 0.5 + a/2, iy*a - 0.5 + a, iz*a - 0.5 + a/2])
                            if iz < 3:
                                fcc_points.append([ix*a - 0.5 + a/2, iy*a - 0.5 + a/2, iz*a - 0.5 + a])
                fcc_points = np.array(fcc_points[:100])
                c_init = fcc_points + rng.normal(0, 0.02, fcc_points.shape)
            elif restart == 2:
                # BCC-like lattice
                a = 0.25
                bcc_points = []
                for ix in range(4):
                    for iy in range(4):
                        for iz in range(4):
                            bcc_points.append([ix*a - 0.5 + a/2, iy*a - 0.5 + a/2, iz*a - 0.5 + a/2])
                            bcc_points.append([ix*a - 0.5 + a/2 + a/2, iy*a - 0.5 + a/2 + a/2, iz*a - 0.5 + a/2 + a/2])
                bcc_points = np.array(bcc_points[:100])
                c_init = bcc_points + rng.normal(0, 0.02, bcc_points.shape)
            else:
                # Random with some structure
                c_init = rng.uniform(-0.45, 0.45, (100, 3))
                # Add small perturbation to break symmetry
                c_init += rng.normal(0, 0.01, (100, 3))
            
            c_init = np.clip(c_init, -0.49, 0.49)
            r_init = compute_actual_radius(c_init)
            
            x0 = np.concatenate([c_init.flatten(), [r_init]])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, 
                               options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False})
                
                if np.all(np.isfinite(res.x)):
                    c_opt = res.x[:-1].reshape(-1, 3)
                    actual_r = compute_actual_radius(c_opt)
                    
                    if actual_r > best_r:
                        best_r = actual_r
                        best_c = c_opt
            except Exception:
                pass

    # Final polish: high-precision local optimization from best solution
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        x0 = np.concatenate([best_c.flatten(), [best_r]])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, bounds=bounds, 
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
            
            if np.all(np.isfinite(res.x)):
                c_opt = res.x[:-1].reshape(-1, 3)
                actual_r = compute_actual_radius(c_opt)
                
                if actual_r > best_r:
                    best_r = actual_r
                    best_c = c_opt
        except Exception:
            pass

    return best_c

if __name__ == "__main__":
    construct()
