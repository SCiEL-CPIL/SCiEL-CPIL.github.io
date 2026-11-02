"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Initial Configuration: Jittered Grid
    # 5x5x4 grid fits 100 spheres
    m = 5
    n_z = 4
    # Create grid points
    x = np.linspace(-0.5, 0.5, m, endpoint=False) + 0.5/m
    y = np.linspace(-0.5, 0.5, m, endpoint=False) + 0.5/m
    z = np.linspace(-0.5, 0.5, n_z, endpoint=False) + 0.5/n_z
    
    # Meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    centers = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to valid range (roughly)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_r = 0.0
    
    # Helper to compute radius
    def compute_radius(c):
        # Pairwise distances
        # Use broadcasting
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        # c is (N, 3). Walls at +/- 0.5
        # dist to wall for coord k is 0.5 - |c[:,k]|
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    # Initial radius
    current_r = compute_radius(best_centers)
    best_r = current_r

    # Binary Search for max r
    # Range: [0.1, 0.12]
    lo, hi = 0.10, 0.12
    
    # Smooth max function: f(x) = 0.5 * (x + sqrt(x^2 + delta^2))
    # We want to minimize sum of smooth_max(0, violation)^2
    # violation_pair = 2*r - dist_ij
    # violation_wall = |c_ik| - (0.5 - r)
    
    def smooth_max(x, delta=0.001):
        return 0.5 * (x + np.sqrt(x**2 + delta**2))
    
    def smooth_max_grad(x, delta=0.001):
        # d/dx [0.5(x + sqrt(x^2+delta^2))] = 0.5(1 + x/sqrt(x^2+delta^2))
        return 0.5 * (1 + x / np.sqrt(x**2 + delta**2))

    def objective_and_grad(c, r, delta=0.001):
        # c is (N, 3)
        # Flatten for optimizer
        # We will return objective and gradient
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero in gradient
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        # Violations: 2r - dist
        viol_p = 2.0 * r - dists
        # Zero out diagonal
        viol_p[np.diag_indices_from(viol_p)] = 0.0
        
        # Smooth max of violations
        sm_p = smooth_max(viol_p, delta)
        # Objective contribution: sum(sm_p^2)
        obj_p = np.sum(sm_p**2)
        
        # Gradient w.r.t c
        # d(obj_p)/d(c_i) = sum_j 2 * sm_p_ij * sm_grad_p_ij * d(2r - dist_ij)/d(c_i)
        # d(2r - dist_ij)/d(c_i) = - (c_i - c_j) / dist_ij = (c_j - c_i) / dist_ij
        # Note: dist is symmetric.
        
        # We need to sum over j != i
        # grad_p = zeros (N, 3)
        # For each i, sum over j: 2 * sm_p_ij * sm_grad_p_ij * (c_j - c_i) / dists_ij
        
        # Vectorized:
        # sm_p is (N, N)
        # sm_grad_p is (N, N)
        # We need (c_j - c_i) / dists_ij
        # Let's compute the term T_ij = 2 * sm_p_ij * sm_grad_p_ij / dists_ij
        # Then grad_p[i] = sum_j T_ij * (c_j - c_i)
        
        # Be careful with diagonal (dist=0). We set dists_safe.
        # But viol_p on diagonal is 0, so sm_p is small, sm_grad is 0.5.
        # The term (c_j - c_i) is 0. So it's fine.
        
        T = 2.0 * sm_p * smooth_max_grad(viol_p, delta) / dists_safe
        # Set diagonal to 0
        T[np.diag_indices_from(T)] = 0.0
        
        # grad_p[i] = sum_j T_ij * (c_j - c_i)
        # = (T @ c) - c * row_sum_T
        # T @ c is (N, 3)
        # row_sum_T is (N, 1)
        
        Tc = T @ c
        row_sum_T = T.sum(axis=1, keepdims=True)
        grad_p = Tc - c * row_sum_T
        
        # Wall Violations
        # |c_ik| - (0.5 - r)
        # Let u_ik = c_ik - (0.5 - r) for c_ik > 0
        # Let v_ik = -c_ik - (0.5 - r) for c_ik < 0
        # Actually simpler: viol_w = |c_ik| - (0.5 - r)
        # smooth_max(viol_w)
        
        abs_c = np.abs(c)
        viol_w = abs_c - (0.5 - r)
        sm_w = smooth_max(viol_w, delta)
        obj_w = np.sum(sm_w**2)
        
        # Gradient w.r.t c_ik
        # d(sm_w^2)/d(c_ik) = 2 * sm_w * sm_grad_w * d(|c_ik|)/d(c_ik)
        # d(|c_ik|)/d(c_ik) = sign(c_ik)
        
        grad_w = 2.0 * sm_w * smooth_max_grad(viol_w, delta) * np.sign(c)
        
        total_obj = obj_p + obj_w
        total_grad = grad_p + grad_w
        
        return total_obj, total_grad

    # Binary Search Loop
    for step in range(40):
        if time.time() > deadline:
            break
            
        mid = (lo + hi) / 2.0
        
        # Try to find a configuration for radius mid
        # Use multiple restarts if time permits, but let's try 1 first
        # We start from the best_centers found so far, or a new jittered grid
        
        # Strategy: Use the current best_centers as initial guess.
        # If it fails, try a random jittered grid.
        
        found = False
        best_c_for_mid = None
        best_obj_for_mid = np.inf
        
        # Try 3 initializations
        inits = [
            best_centers + np.random.normal(0, 0.005, best_centers.shape),
            best_centers + np.random.normal(0, 0.01, best_centers.shape),
            # Fresh jittered grid
            (np.random.rand(N, 3) - 0.5) * 0.9
        ]
        
        for init_c in inits:
            if time.time() > deadline:
                break
                
            # Clip to ensure we start inside
            init_c = np.clip(init_c, -0.49, 0.49)
            
            # Flatten
            c_flat = init_c.flatten()
            
            # Objective function for scipy
            def obj_func(x):
                c = x.reshape(N, 3)
                val, _ = objective_and_grad(c, mid, delta=0.001)
                return val
            
            # Gradient function
            def grad_func(x):
                c = x.reshape(N, 3)
                _, g = objective_and_grad(c, mid, delta=0.001)
                return g.flatten()
            
            try:
                res = minimize(
                    obj_func, 
                    c_flat, 
                    jac=grad_func, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                
                c_opt = res.x.reshape(N, 3)
                obj_val = res.fun
                
                if obj_val < best_obj_for_mid:
                    best_obj_for_mid = obj_val
                    best_c_for_mid = c_opt
                    
                # If objective is very small, we consider it feasible
                if obj_val < 1e-6:
                    found = True
                    break
                    
            except Exception:
                continue
        
        if found:
            # mid is feasible
            lo = mid
            best_centers = best_c_for_mid
            best_r = compute_radius(best_centers)
            # Update best_r with the actual computed radius, which might be slightly higher than mid due to tolerance
            if best_r < mid:
                # This shouldn't happen if feasible, but just in case
                best_r = mid
        else:
            # mid is not feasible
            hi = mid
            
    # Final Polish
    # Run a final optimization with the best_r found to maximize the radius further?
    # Or just return best_centers.
    # The binary search converges to the limit.
    
    # Ensure validity
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = 5
        n_z = 4
        x = np.linspace(-0.5, 0.5, m, endpoint=False) + 0.5/m
        y = np.linspace(-0.5, 0.5, m, endpoint=False) + 0.5/m
        z = np.linspace(-0.5, 0.5, n_z, endpoint=False) + 0.5/n_z
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        best_centers = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        
    return best_centers
# EVOLVE-BLOCK-END
