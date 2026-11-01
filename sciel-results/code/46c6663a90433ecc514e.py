"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s, safety margin
    
    best_result = None
    best_score = -np.inf
    
    def penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Boundary penalty
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)
        
        # Overlap penalty (vectorized)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        # Only consider upper triangle to avoid double counting and self
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlap_viol = np.sum(np.maximum(0, min_dist[mask] - dist[mask])**2)
        
        penalty = boundary_viol + overlap_viol
        return penalty
    
    def objective(params):
        n = N
        r = params[2*n:]
        pen = penalty_function(params)
        # We want to maximize sum(r), so minimize -sum(r) + lambda * penalty
        # Use a large lambda to strongly penalize violations
        return -np.sum(r) + 1000.0 * pen
    
    def is_valid(params, tol=1e-7):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Check boundaries
        if np.any(xy[:, 0] < r - tol) or np.any(xy[:, 0] > 1 - r + tol):
            return False
        if np.any(xy[:, 1] < r - tol) or np.any(xy[:, 1] > 1 - r + tol):
            return False
        
        # Check overlaps
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if np.any(dist[mask] < min_dist[mask] - tol):
            return False
            
        return True
    
    def get_score(params):
        n = N
        r = params[2*n:]
        return np.sum(r)
    
    # Initial setup: grid-based
    side = 6
    x0 = np.zeros(N * 3)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx * 3] = (i + 0.5) / side
            x0[idx * 3 + 1] = (j + 0.5) / side
            x0[idx * 3 + 2] = 0.5 / side - 1e-4
            idx += 1
    
    # Reshape to (2N + N) format: xy interleaved then r? 
    # My objective expects params[:2*n] as xy (flattened) and params[2*n:] as r
    # So x0 needs to be rearranged.
    # Current x0: [x0, y0, r0, x1, y1, r1, ...]
    # Target: [x0, x1, ..., xN-1, y0, y1, ..., yN-1, r0, r1, ..., rN-1]
    
    def rearrange_params(params):
        # Input: [x0, y0, r0, x1, y1, r1, ...]
        # Output: [x0, ..., xN, y0, ..., yN, r0, ..., rN]
        n = N
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        return np.concatenate([x, y, r])
    
    def unarrange_params(params):
        # Input: [x0, ..., xN, y0, ..., yN, r0, ..., rN]
        # Output: [x0, y0, r0, x1, y1, r1, ...]
        n = N
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        result = np.zeros(n * 3)
        result[0::3] = x
        result[1::3] = y
        result[2::3] = r
        return result

    x0_flat = rearrange_params(x0)
    
    best_result_flat = x0_flat.copy()
    best_score = get_score(x0_flat)
    
    # Try a few restarts with slight perturbations
    np.random.seed(42)
    
    for restart in range(8):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_current = x0_flat.copy()
        else:
            x_current = x0_flat.copy()
            # Add small random noise to positions and radii
            noise_pos = np.random.normal(0, 0.02, N)
            noise_rad = np.random.normal(0, 0.005, N)
            x_current[0:N] += noise_pos
            x_current[N:2*N] += noise_pos
            x_current[2*N:] += noise_rad
            
        try:
            res = scipy.optimize.minimize(objective, x_current, method='L-BFGS-B', 
                                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            x_opt = res.x
            
            # Check validity
            if is_valid(x_opt):
                actual_value = get_score(x_opt)
                if actual_value > best_score:
                    best_score = actual_value
                    best_result_flat = x_opt.copy()
                    
        except Exception:
            pass

    # Final polish on the best result
    if time.time() < deadline - 10:
        try:
            def objective_polish(params):
                n = N
                r = params[2*n:]
                pen = penalty_function(params)
                return -np.sum(r) + 10000.0 * pen

            res = scipy.optimize.minimize(objective_polish, best_result_flat, method='L-BFGS-B', 
                                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            x_opt = res.x
            
            if is_valid(x_opt):
                actual_value = get_score(x_opt)
                if actual_value > best_score:
                    best_score = actual_value
                    best_result_flat = x_opt.copy()
                    
        except Exception:
            pass

    # Format output
    result_flat = unarrange_params(best_result_flat)
    result = np.zeros((N, 3))
    result[:, 0] = result_flat[0::3]
    result[:, 1] = result_flat[1::3]
    result[:, 2] = result_flat[2::3]
    
    return result

# EVOLVE-BLOCK-END
