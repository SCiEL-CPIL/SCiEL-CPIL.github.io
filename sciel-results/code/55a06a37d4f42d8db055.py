import numpy as np
import time
import math

N = 150

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave 20s buffer for the 120s limit
    
    # 1. Generate initial Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates (theta, phi)
    # theta: [0, pi], phi: [0, 2pi]
    # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
    theta_init = np.arccos(x_init[:, 2])
    phi_init = np.arctan2(x_init[:, 1], x_init[:, 0])
    # Ensure phi is in [0, 2pi)
    phi_init = phi_init % (2 * np.pi)
    
    # Flatten to 1D array for optimizer
    x0 = np.column_stack([theta_init, phi_init]).flatten()
    
    # Define objective function
    def objective(v):
        if not np.all(np.isfinite(v)):
            return 1e100
        th = v[0::2]
        ph = v[1::2]
        
        # Convert to Cartesian
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        pts = np.column_stack([x, y, z])
        
        # Normalize just in case of numerical drift
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Calculate Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
        # Vectorized distance calculation
        # To avoid i==j, we can compute full matrix and zero out diagonal
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Avoid log(0)
        dist_safe = np.where(mask, dist, 1.0)
        
        # Sum over pairs
        # We sum over all i != j, then divide by 2
        log_dists = np.log(dist_safe)
        energy = -0.5 * np.sum(log_dists[mask])
        
        return energy
    
    # Use L-BFGS-B
    from scipy.optimize import minimize
    
    best_energy = objective(x0)
    best_x = x0
    
    try:
        # Use L-BFGS-B, max_iter can be high, but we check time
        # finite_diff_step can be small
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_energy:
            if np.all(np.isfinite(res.x)):
                best_x = res.x
                best_energy = res.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_limit:
        # Try a second pass with a different initial perturbation if time allows?
        # Or just return the best found.
        # Let's try a quick local refinement with gradient descent if L-BFGS didn't converge perfectly?
        # L-BFGS is usually sufficient.
        
        # Let's try one more restart with a small random perturbation if time is abundant
        if time.time() - start_time < 80:
            # Perturb initial points slightly
            rng = np.random.default_rng(42)
            theta_pert = theta_init + rng.normal(0, 0.01, N)
            phi_pert = phi_init + rng.normal(0, 0.01, N)
            # Clip theta
            theta_pert = np.clip(theta_pert, 1e-6, np.pi - 1e-6)
            x0_pert = np.column_stack([theta_pert, phi_pert]).flatten()
            
            try:
                res2 = minimize(objective, x0_pert, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
                if res2.success or res2.fun < best_energy:
                    if np.all(np.isfinite(res2.x)):
                        best_x = res2.x
                        best_energy = res2.fun
            except Exception:
                pass

    # Convert best_x back to Cartesian
    th = best_x[0::2]
    ph = best_x[1::2]
    
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    pts = np.column_stack([x, y, z])
    
    # Final normalization
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Check for duplicates or invalid values
    if not np.all(np.isfinite(pts)):
        # Fallback to initial
        return x_init
        
    # Verify no points are too close
    # This is a sanity check, the optimizer should handle it
    return pts
