import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_sum = 0.0
    best_solution = None
    
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
                
                # Overlap constraints
                # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
                # dist >= ri + rj
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        req = r[i] + r[j]
                        if dist < req:
                            pen += (req - dist)**2
                
                # We want to maximize sum_r, so minimize -sum_r + lambda * pen
                # Lambda needs to be large enough to enforce constraints
                lambda_pen = 10.0
                return -sum_r + lambda_pen * pen
            
            # Use L-BFGS-B or BFGS
            # BFGS is unconstrained, which fits the penalty method
            res = minimize(objective, v0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-8})
            
            if res.success or res.fun < 0: # res.fun < 0 means sum_r > 0
                v_res = res.x
                x_res = v_res[:N]
                y_res = v_res[N:2*N]
                r_res = v_res[2*N:]
                
                # Check validity
                valid = True
                for i in range(N):
                    if r_res[i] <= 0:
                        valid = False
                        break
                    if x_res[i] + r_res[i] > W + 1e-6 or x_res[i] - r_res[i] < -1e-6:
                        valid = False
                        break
                    if y_res[i] + r_res[i] > H + 1e-6 or y_res[i] - r_res[i] < -1e-6:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x_res[i] - x_res[j]
                            dy = y_res[i] - y_res[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_res[i] + r_res[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    sum_r = np.sum(r_res)
                    if sum_r > best_sum:
                        best_sum = sum_r
                        # Store solution in a standard format? 
                        # The task asks for an array of shape (n, 3).
                        # It doesn't specify which rectangle dimensions to use in the output, 
                        # but the evaluator likely checks the bounding box of the returned circles.
                        # The bounding box of the circles will have width 2*max(r) if centered? No.
                        # The evaluator measures "raw value". 
                        # The task says "bounding rectangle has perimeter at most 4".
                        # If we return circles, the evaluator will compute the min/max x/y to find the bounding box.
                        # We must ensure the bounding box of the returned circles has perimeter <= 4.
                        # Our optimization assumed a fixed W, H with 2(W+H)=4.
                        # The actual bounding box of the circles might be smaller than W, H.
                        # If it is smaller, the perimeter is < 4, which is allowed.
                        # So we can just return the circles.
                        
                        # However, we should probably scale them to fill the perimeter 4?
                        # No, the objective is to maximize sum of radii.
                        # If we have a valid packing in W, H, the sum of radii is fixed.
                        # If we scale it up, we might violate the perimeter constraint if the bounding box touches the edges.
                        # But if the bounding box is strictly inside, we could scale up.
                        # However, the optimization usually pushes circles to the boundaries to maximize radii.
                        # So the bounding box will likely be close to W, H.
                        
                        # Let's just record the solution.
                        best_solution = np.column_stack([x_res, y_res, r_res])
                        
    # If no valid solution found, return a fallback
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Scale to fit perimeter 4?
        # The seed program returned these coordinates. 
        # The bounding box of these points is roughly 1x1 (if side=5, range 0.1 to 0.9).
        # Perimeter of 1x1 box is 4.
        # So the seed program works.
        return np.array(cells[:N])
    
    return best_solution
