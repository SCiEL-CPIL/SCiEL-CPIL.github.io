import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    def objective(x):
        # x is a flattened array of 16 floats (8 points, 2 coords each)
        points = x.reshape(8, 2)
        
        # Compute pairwise squared distances
        # diff[i, j, k] = points[i, k] - points[j, k]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        
        # Extract upper triangle indices
        i, j = np.triu_indices(8, k=1)
        d2 = sq_dists[i, j]
        
        # Check for zero distances
        if np.any(d2 < 1e-12):
            return 1e10
            
        # D = max(d) = sqrt(max(d2))
        max_d2 = np.max(d2)
        if max_d2 < 1e-12:
            return 1e10
            
        # Objective: - sum(log(d_ij)) + 28 * log(D)
        # log(d_ij) = 0.5 * log(d2_ij)
        # log(D) = 0.5 * log(max_d2)
        
        sum_log_d = 0.5 * np.sum(np.log(d2))
        log_D = 0.5 * np.log(max_d2)
        
        # We want to maximize sum(log(d_ij)) - 28*log(D)
        # So minimize: -sum(log(d_ij)) + 28*log(D)
        return -(sum_log_d) + 28.0 * log_D

    def gradient(x):
        # Numerical gradient is sufficient for 16 dims, but let's stick to finite differences in minimize
        # L-BFGS-B uses numerical gradients by default if not provided
        pass

    # Initial guesses
    inits = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p_reg.flatten())
    
    # 2. Perturbed Regular Octagons
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise = rng.normal(0, 0.1, (8, 2))
        p_pert = p_reg + noise
        inits.append(p_pert.flatten())
        
    # 3. Two Concentric Squares
    # Square 1: angles 0, pi/2, pi, 3pi/2
    # Square 2: angles pi/4, 3pi/4, 5pi/4, 7pi/4
    # Try different radii
    for r1 in [0.5, 0.8, 1.0, 1.2]:
        for r2 in [0.5, 0.8, 1.0, 1.2]:
            if r1 == r2: continue
            a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            a2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
            p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
            p_two = np.vstack([p1, p2])
            inits.append(p_two.flatten())
            
    # 4. Deformed Octagon (varying angles)
    for _ in range(5):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        # Add some structured perturbation to angles
        pert = np.sin(angles * 2) * 0.2
        angles_pert = angles + pert
        p_def = np.column_stack([np.cos(angles_pert), np.sin(angles_pert)])
        inits.append(p_def.flatten())

    best_x = inits[0]
    best_val = objective(best_x)
    
    # Optimization Loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Use L-BFGS-B for speed and robustness
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
                # Optional: Polish with Nelder-Mead for extra precision
                if time.time() < deadline - 5.0:
                    res2 = minimize(
                        objective, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_x = res2.x
                        
        except Exception:
            pass

    best_points = best_x.reshape(8, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
