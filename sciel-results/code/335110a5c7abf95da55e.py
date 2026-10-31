import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Initial configuration: 5x5 grid with small offset for optimization
    side = 5
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            positions[idx] = [(i + 0.5) / side, (j + 0.5) / side]
            radii[idx] = 0.5 / side - 0.001
    
    # Objective: maximize sum of radii (minimize negative sum)
    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        rad = params[2 * N:]
        
        # Penalty for going outside [0,1]^2
        penalty = 0.0
        for i in range(N):
            x, y = pos[i]
            r = rad[i]
            if x - r < 0:
                penalty += (x - r) ** 2
            if x + r > 1:
                penalty += (x + r - 1) ** 2
            if y - r < 0:
                penalty += (y - r) ** 2
            if y + r > 1:
                penalty += (y + r - 1) ** 2
        
        # Penalty for overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return -np.sum(rad) + 100.0 * penalty
    
    # Run multiple restarts with different initial configurations
    best_result = None
    best_value = -np.inf
    
    # Try several initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 30:
            break
        
        # Create perturbed initial configuration
        np.random.seed(restart * 42)
        pos_init = positions.copy()
        rad_init = radii.copy()
        
        # Add small random perturbations
        pos_init += np.random.normal(0, 0.01, (N, 2))
        rad_init += np.random.normal(0, 0.001, N)
        
        # Ensure radii are positive
        rad_init = np.maximum(rad_init, 0.01)
        
        # Clip positions to stay within bounds
        pos_init[:, 0] = np.clip(pos_init[:, 0], 0.05, 0.95)
        pos_init[:, 1] = np.clip(pos_init[:, 1], 0.05, 0.95)
        
        params_init = np.concatenate([pos_init.flatten(), rad_init])
        
        try:
            result = minimize(objective, params_init, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() - start_time > deadline - 10:
                break
            
            if result.success or result.fun < best_value:
                pos_opt = result.x[:2 * N].reshape(N, 2)
                rad_opt = result.x[2 * N:]
                
                # Verify validity
                valid = True
                for i in range(N):
                    x, y = pos_opt[i]
                    r = rad_opt[i]
                    if r <= 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                        valid = False
                        break
                    if valid:
                        for j in range(i + 1, N):
                            dist = np.linalg.norm(pos_opt[i] - pos_opt[j])
                            if dist < rad_opt[i] + rad_opt[j] - 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    value = np.sum(rad_opt)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([pos_opt, rad_opt])
        
        except Exception:
            continue
    
    # If no valid result from optimization, fall back to seed
    if best_result is None:
        return np.array([[positions[i, 0], positions[i, 1], radii[i]] for i in range(N)])
    
    return best_result
