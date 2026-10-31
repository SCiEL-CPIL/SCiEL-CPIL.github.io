import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial Guess: 5x5 grid + 1 center
    # 5x5 grid centers: 0.1, 0.3, 0.5, 0.7, 0.9
    # Radius for 5x5 grid: 0.1 is max touching neighbors. Let's start with 0.09.
    
    init_x = []
    init_y = []
    init_r = []
    
    for i in range(5):
        for j in range(5):
            init_x.append(0.1 + 0.2 * i)
            init_y.append(0.1 + 0.2 * j)
            init_r.append(0.09)
            
    # 26th circle at center
    init_x.append(0.5)
    init_y.append(0.5)
    init_r.append(0.01)
    
    x0 = np.array(init_x + init_y + init_r)
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    x0[:2*N] += np.random.uniform(-0.01, 0.01, 2*N)
    x0[2*N:] = np.maximum(x0[2*N:], 0.01) # Ensure positive radii
    
    # Define constraints
    def constraints(x):
        c = []
        x_coords = x[:N]
        y_coords = x[N:2*N]
        radii = x[2*N:]
        
        # Boundary constraints
        c.append(x_coords - radii)
        c.append(1 - x_coords - radii)
        c.append(y_coords - radii)
        c.append(1 - y_coords - radii)
        
        # Positivity
        c.append(radii - 1e-6)
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist_sq = dx*dx + dy*dy
                sum_r = radii[i] + radii[j]
                c.append(dist_sq - sum_r*sum_r)
                
        return c
        
    # Objective: minimize negative sum of radii
    def objective(x):
        radii = x[2*N:]
        return -np.sum(radii)
        
    # Bounds
    # x, y in [0, 1], r in [0, 0.5] (max possible radius is 0.5 if one circle, but here smaller)
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((1e-6, 0.5)) # r
        
    # Run optimization
    best_x = x0
    best_val = objective(x0)
    
    # SLSQP is good for this size
    # We might try multiple restarts if time permits
    # For now, one robust run with tight tolerances
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'ineq', 'fun': constraints},
                       options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
        
        if res.success:
            best_x = res.x
            best_val = res.fun
        else:
            # If it didn't converge perfectly, maybe the best found during iterations is better?
            # SLSQP doesn't easily expose intermediate bests in this API without callback.
            # We'll assume res.x is the best effort.
            if objective(res.x) < best_val:
                best_x = res.x
                best_val = res.fun
                
    except Exception as e:
        pass
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try a second run with a different perturbation if we have time
        x0_2 = x0.copy()
        x0_2[:2*N] += np.random.uniform(-0.05, 0.05, 2*N)
        x0_2[2*N:] = np.maximum(x0_2[2*N:], 0.01)
        
        try:
            res2 = minimize(objective, x0_2, method='SLSQP', bounds=bounds, 
                            constraints={'type': 'ineq', 'fun': constraints},
                            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if res2.success and res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
            elif not res2.success and objective(res2.x) < best_val:
                best_x = res2.x
                best_val = objective(res2.x)
                
        except Exception as e:
            pass
            
    # Construct output array (x, y, r)
    x_coords = best_x[:N]
    y_coords = best_x[N:2*N]
    radii = best_x[2*N:]
    
    # Final check for validity and clipping
    # Ensure circles are inside [0,1]x[0,1]
    # If a circle is slightly out due to numerical error, clip center or reduce radius?
    # The constraints should have prevented this, but let's be safe.
    for i in range(N):
        # Clip radii to ensure boundary satisfaction if numerical errors occurred
        max_r = min(x_coords[i], 1 - x_coords[i], y_coords[i], 1 - y_coords[i])
        if radii[i] > max_r:
            radii[i] = max(max_r, 1e-6)
            
    # Check pairwise distances for final validity (and fix if slightly violated)
    # This is a post-processing step to ensure the returned array is strictly valid
    # If violation, reduce radius of the smaller circle or both?
    # Simple approach: if dist < r_i + r_j, scale down radii proportionally or just set r_j = dist - r_i - epsilon
    # This might be overkill if SLSQP worked, but ensures score 0.0 doesn't happen due to tiny epsilon violations.
    
    for i in range(N):
        for j in range(i+1, N):
            dx = x_coords[i] - x_coords[j]
            dy = y_coords[i] - y_coords[j]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                # Overlap detected. Reduce the larger radius? Or both?
                # To maximize sum, we should reduce the "excess" as little as possible.
                # Let r_i' + r_j' = dist - 1e-9
                # Keep ratio? Or just reduce the one that allows more?
                # Simple: reduce both by half the excess
                excess = sum_r - dist
                if excess > 0:
                    radii[i] -= excess / 2
                    radii[j] -= excess / 2
                    # Ensure positivity
                    radii[i] = max(radii[i], 1e-6)
                    radii[j] = max(radii[j], 1e-6)
                    
    # Re-check boundaries after radius reduction?
    # Radius reduction only helps boundaries.
    
    return np.column_stack((x_coords, y_coords, radii))
