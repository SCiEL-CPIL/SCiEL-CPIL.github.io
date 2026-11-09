import numpy as np
import time
from scipy.optimize import minimize

N = 150

def construct():
        start_time = time.time()
        deadline = start_time + 140.0 # 10s margin

        # 1. Initial points: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Flatten for optimizer
        x_flat_init = x_init.flatten()

        # 2. Objective function
        def objective(x_flat):
            x = x_flat.reshape(N, 3)
            # Distances
            # Compute pairwise distances efficiently
            # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            # dist = np.linalg.norm(diff, axis=2)
            
            # To avoid division by zero and log(0), add small epsilon
            # But points shouldn't be that close.
            
            # Energy calculation
            # Sum over i < j
            # We can compute full distance matrix and take upper triangle
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0) # Avoid self distance
            # Only sum i < j
            iu = np.triu_indices(N, k=1)
            d = dist[iu]
            E_log = np.sum(np.log(1.0 / d))
            
            # Penalty for leaving sphere
            norms = np.linalg.norm(x, axis=1)
            E_pen = 100.0 * np.sum((norms - 1.0)**2)
            
            return E_log + E_pen

        def gradient(x_flat):
            x = x_flat.reshape(N, 3)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            
            # Gradient of log(1/d) wrt x_i is (x_i - x_j) / d^2
            # Sum over j != i
            # grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
            # Note: dist_sq is |x_i - x_j|^2
            # We need to sum (x_i - x_j) / dist_sq_ij
            # This is equivalent to summing vectors weighted by 1/dist_sq
            
            # Create a mask for i != j
            mask = ~np.eye(N, dtype=bool)
            
            # Weighted sum
            # weights[i, j] = 1 / dist_sq[i, j] if i != j else 0
            weights = np.zeros_like(dist_sq)
            weights[mask] = 1.0 / dist_sq[mask]
            
            # grad_i = sum_j weights[i, j] * (x_i - x_j)
            # = sum_j weights[i, j] * x_i - sum_j weights[i, j] * x_j
            # = x_i * sum_j weights[i, j] - sum_j (weights[i, j] * x_j)
            
            sum_weights = np.sum(weights, axis=1) # shape (N,)
            # weighted_x = weights @ x  (shape (N, 3))
            weighted_x = weights @ x
            
            grad = x * sum_weights[:, np.newaxis] - weighted_x
            
            # Gradient of penalty
            norms = np.linalg.norm(x, axis=1)
            # d/dx_i (norms_i - 1)^2 = 2 (norms_i - 1) * (x_i / norms_i)
            # = 2 x_i (1 - 1/norms_i)
            # Wait, penalty is (|x|^2 - 1)^2? No, I used (norm - 1)^2.
            # Let's use (|x|^2 - 1)^2 for simpler gradient?
            # d/dx (|x|^2 - 1)^2 = 2 (|x|^2 - 1) * 2x = 4x (|x|^2 - 1)
            # Let's stick to (norm - 1)^2.
            # grad_pen_i = 2 * (norms_i - 1) * (x_i / norms_i)
            # Be careful if norms_i is 0.
            safe_norms = np.where(norms < 1e-6, 1.0, norms)
            grad_pen = 2.0 * 100.0 * (norms - 1.0) / safe_norms * x
            
            return (grad + grad_pen).flatten()

        # 3. Optimize
        best_x = x_flat_init
        best_energy = objective(best_x)
        
        # Use L-BFGS-B
        # We might need multiple restarts or just one good run.
        # Given time constraints, one run with L-BFGS-B is usually sufficient to beat Fibonacci.
        
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_energy:
                best_x = res.x
                best_energy = res.fun
        except Exception:
            pass

        # Check time
        if time.time() < deadline:
            # Maybe try a second pass with tighter tolerance or different start?
            # For now, normalize and return.
            pass

        x_final = best_x.reshape(N, 3)
        # Normalize to ensure they are on the unit sphere
        norms = np.linalg.norm(x_final, axis=1)
        x_final = x_final / norms[:, np.newaxis]
        
        return x_final
