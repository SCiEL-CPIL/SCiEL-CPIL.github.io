"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer for overhead
    
    # Initial grid setup
    # 32 circles. A 6x6 grid has 36 spots. We use the first 32.
    # Or a 5x7 grid? 5x7=35.
    # Let's stick to 6x6 for simplicity, or maybe 4x8?
    # Square aspect ratio is 1:1. 6x6 is closest to square.
    side = 6
    x0 = np.zeros(N * 3)
    
    # Generate grid points
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx * 3] = (i + 0.5) / side
            x0[idx * 3 + 1] = (j + 0.5) / side
            # Initial radius slightly less than half cell width to be safe
            x0[idx * 3 + 2] = 0.5 / side - 1e-4
            idx += 1
    
    best_result = x0.copy()
    best_score = -np.sum(best_result[2::3]) # Maximize sum of radii, so minimize negative sum
    
    # Penalty function
    def objective(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Sum of radii to maximize
        sum_r = np.sum(r)
        
        # Boundary violations
        # x_i >= r_i, x_i <= 1 - r_i
        # y_i >= r_i, y_i <= 1 - r_i
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                        np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        # Overlap violations
        overlap_viol = 0.0
        # N=32, so 32*31/2 = 496 pairs
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_viol += (min_dist - dist)**2
        
        penalty = boundary_viol + overlap_viol
        
        # We want to maximize sum_r, so minimize -sum_r + lambda * penalty
        # Start with a moderate lambda, increase if needed?
        # For L-BFGS-B, a fixed large lambda is often better to ensure feasibility first.
        lambda_val = 100.0
        return -sum_r + lambda_val * penalty
    
    # Gradient (optional, L-BFGS-B can use numerical gradients, but analytical is faster)
    # Given the complexity, let's try numerical first. If too slow, implement gradient.
    # With N=32, 96 variables. Numerical gradient is 96 function calls.
    # Function call is ~500 operations. 96*500 = 48000 ops. Very fast.
    
    def run_optimization(x_init):
        res = scipy.optimize.minimize(objective, x_init, method='L-BFGS-B', 
                                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        return res.x, res.fun
    
    # Try a few restarts with slight perturbations
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_current = x0.copy()
        else:
            x_current = x0.copy()
            # Add small random noise to positions and radii
            noise_pos = np.random.normal(0, 0.01, N)
            noise_rad = np.random.normal(0, 0.001, N)
            x_current[0::3] += noise_pos
            x_current[1::3] += noise_pos
            x_current[2::3] += noise_rad
            
        try:
            x_opt, score = run_optimization(x_current)
            
            # Check validity and improve score tracking
            # The objective is -sum_r + penalty.
            # We want to compare based on -sum_r (actual value) if feasible, or penalized value.
            # For the final output, we need a valid packing.
            # Let's check if the solution is feasible.
            x = x_opt[0::3]
            y = x_opt[1::3]
            r = x_opt[2::3]
            
            is_valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6:
                    is_valid = False
                    break
                if y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                    is_valid = False
                    break
            
            if is_valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        if dist < r[i] + r[j] - 1e-6:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            actual_value = np.sum(r)
            
            if is_valid and actual_value > best_score:
                best_score = actual_value
                best_result = x_opt.copy()
                
        except Exception:
            pass

    # If we still have time, do a final polish on the best result with higher precision
    if time.time() < deadline - 10:
        try:
            def objective_polish(params):
                x = params[0::3]
                y = params[1::3]
                r = params[2::3]
                sum_r = np.sum(r)
                
                viol_x_low = np.maximum(0, r - x)
                viol_x_high = np.maximum(0, r - (1 - x))
                viol_y_low = np.maximum(0, r - y)
                viol_y_high = np.maximum(0, r - (1 - y))
                
                boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                                np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
                
                overlap_viol = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        min_dist = r[i] + r[j]
                        if dist < min_dist:
                            overlap_viol += (min_dist - dist)**2
                
                penalty = boundary_viol + overlap_viol
                lambda_val = 1000.0 # Higher penalty for polish
                return -sum_r + lambda_val * penalty

            res = scipy.optimize.minimize(objective_polish, best_result, method='L-BFGS-B', 
                                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            is_valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-8 or x[i] > 1 - r[i] + 1e-8:
                    is_valid = False
                    break
                if y[i] < r[i] - 1e-8 or y[i] > 1 - r[i] + 1e-8:
                    is_valid = False
                    break
            
            if is_valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        if dist < r[i] + r[j] - 1e-8:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid:
                actual_value = np.sum(r)
                if actual_value > best_score:
                    best_score = actual_value
                    best_result = res.x.copy()
                    
        except Exception:
            pass

    # Format output
    result = np.zeros((N, 3))
    result[:, 0] = best_result[0::3]
    result[:, 1] = best_result[1::3]
    result[:, 2] = best_result[2::3]
    
    return result

# EVOLVE-BLOCK-END
