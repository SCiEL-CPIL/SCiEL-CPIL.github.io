import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_sum = 0.0
    best_solution = None
    
    # Precompute pairwise indices for vectorized overlap computation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Try different aspect ratios
    # Perimeter 4 => 2(W+H) = 4 => W+H = 2
    # Let W = 2 / (1 + k), H = 2k / (1 + k)
    # k = W/H
    # k ranges from 0.5 to 2.0 (symmetry)
    
    aspect_ratios = np.linspace(0.6, 1.6, 15)
    
    for k in aspect_ratios:
        if time.time() > deadline:
            break
            
        W = 2.0 / (1.0 + k)
        H = 2.0 * k / (1.0 + k)
        
        # Multi-start optimization for this aspect ratio
        n_starts = 5
        
        for s in range(n_starts):
            if time.time() > deadline:
                break
                
            # Initialize positions and radii
            # Randomly scatter circles
            np.random.seed(42 + s + int(k*100))
            
            # Initial radii: small
            r = np.full(N, 0.05)
            x = np.random.uniform(r, W - r, N)
            y = np.random.uniform(r, H - r, N)
            
            v0 = np.concatenate([x, y, r])
            
            def objective(v):
                x = v[:N]
                y = v[N:2*N]
                r = v[2*N:]
                
                # Ensure positive radii
                r = np.maximum(r, 1e-6)
                
                # Sum of radii to maximize (so minimize negative)
                sum_r = np.sum(r)
                
                # Penalties
                pen = 0.0
                
                # Boundary constraints
                # x + r <= W
                pen += np.sum(np.maximum(0, x + r - W)**2)
                # x - r >= 0
                pen += np.sum(np.maximum(0, r - x)**2)
                # y + r <= H
                pen += np.sum(np.maximum(0, y + r - H)**2)
                # y - r >= 0
                pen += np.sum(np.maximum(0, r - y)**2)
                
                # Overlap constraints - vectorized
                dx = x[i_idx] - x[j_idx]
                dy = y[i_idx] - y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                req = r[i_idx] + r[j_idx]
                overlap = np.maximum(0, req - dist)
                pen += np.sum(overlap**2)
                
                # We want to maximize sum_r, so minimize -sum_r + lambda * pen
                # Lambda needs to be large enough to enforce constraints
                lambda_pen = 10.0
                return -sum_r + lambda_pen * pen
            
            # Use BFGS for the main optimization
            res = minimize(objective, v0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
            
            if res.success or res.fun < 0:
                v_res = res.x
                x_res = v_res[:N]
                y_res = v_res[N:2*N]
                r_res = v_res[2*N:]
                
                # Check validity
                valid = True
                if np.any(r_res <= 0):
                    valid = False
                elif np.any(x_res + r_res > W + 1e-6) or np.any(x_res - r_res < -1e-6):
                    valid = False
                elif np.any(y_res + r_res > H + 1e-6) or np.any(y_res - r_res < -1e-6):
                    valid = False
                else:
                    # Check overlaps - vectorized
                    dx = x_res[i_idx] - x_res[j_idx]
                    dy = y_res[i_idx] - y_res[j_idx]
                    dist = np.sqrt(dx*dx + dy*dy)
                    req = r_res[i_idx] + r_res[j_idx]
                    if np.any(dist < req - 1e-6):
                        valid = False
                
                if valid:
                    # Final polish pass with tighter tolerance
                    if time.time() < deadline - 5:
                        res2 = minimize(objective, v_res, method='BFGS', 
                                       options={'maxiter': 200, 'gtol': 1e-10})
                        v_res = res2.x
                        x_res = v_res[:N]
                        y_res = v_res[N:2*N]
                        r_res = v_res[2*N:]
                        
                        # Re-check validity after polish
                        valid = True
                        if np.any(r_res <= 0):
                            valid = False
                        elif np.any(x_res + r_res > W + 1e-6) or np.any(x_res - r_res < -1e-6):
                            valid = False
                        elif np.any(y_res + r_res > H + 1e-6) or np.any(y_res - r_res < -1e-6):
                            valid = False
                        else:
                            dx = x_res[i_idx] - x_res[j_idx]
                            dy = y_res[i_idx] - y_res[j_idx]
                            dist = np.sqrt(dx*dx + dy*dy)
                            req = r_res[i_idx] + r_res[j_idx]
                            if np.any(dist < req - 1e-6):
                                valid = False
                    
                    if valid:
                        sum_r = np.sum(r_res)
                        if sum_r > best_sum:
                            best_sum = sum_r
                            best_solution = np.column_stack([x_res, y_res, r_res])
                        
    # If no valid solution found, return a fallback
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
