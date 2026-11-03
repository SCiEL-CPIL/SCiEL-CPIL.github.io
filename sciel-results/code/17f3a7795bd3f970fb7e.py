import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # Define the objective function
    def objective(v):
        # v = [x1, y1, r1, ..., x21, y21, r21, W]
        W = v[-1]
        H = 2.0 - W
        
        # Extract positions and radii
        coords = v[:-1].reshape(N, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        rs = coords[:, 2]
        
        sum_r = np.sum(rs)
        
        penalty = 0.0
        
        # Boundary penalties
        # x in [r, W-r]
        penalty += np.sum(np.maximum(0, rs - xs))
        penalty += np.sum(np.maximum(0, xs + rs - W))
        # y in [r, H-r]
        penalty += np.sum(np.maximum(0, rs - ys))
        penalty += np.sum(np.maximum(0, ys + rs - H))
        
        # Overlap penalties
        # dist(i,j) >= ri + rj
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    penalty += overlap
        
        return -sum_r + 100.0 * penalty
    
    # Initial configurations
    def get_initial(W_init):
        H_init = 2.0 - W_init
        # Try a grid layout
        # 5x5 grid
        side = 5
        xs = []
        ys = []
        rs = []
        # We want to fit 21 circles.
        # Let's place them in a 5x5 grid, skipping the last 4.
        # Or maybe a 4x6 grid?
        # Let's stick to 5x5 for simplicity, centered.
        for i in range(side):
            for j in range(side):
                if len(xs) < N:
                    xs.append((i + 0.5) / side)
                    ys.append((j + 0.5) / side)
                    rs.append(0.5 / side)
        
        # Scale to fit W_init, H_init
        # Current bounding box is 1x1.
        # We want to fit in W_init x H_init.
        # We can just let the optimizer adjust W.
        # But for a good start, let's set W = W_init.
        # The positions are in [0,1].
        # We should scale them to [0, W_init] and [0, H_init]?
        # No, the objective function handles the scaling via W and H.
        # But the initial radii should be reasonable.
        # If W_init is 1, H_init is 1, radii 0.1 is fine.
        # If W_init is 2, H_init is 0, that's bad.
        # Let's try a few aspect ratios.
        
        v = np.zeros(N*3 + 1)
        for i in range(N):
            v[3*i] = xs[i] * W_init
            v[3*i+1] = ys[i] * H_init
            v[3*i+2] = 0.5 / side * min(W_init, H_init) # Approximate
        v[-1] = W_init
        return v

    # Try a few initial aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        W_init = ar / (1 + ar) * 2.0
        v0 = get_initial(W_init)
        
        # Run optimization
        try:
            res = minimize(objective, v0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if time.time() > deadline:
                break
                
            # Check if the solution is valid
            v_res = res.x
            W = v_res[-1]
            H = 2.0 - W
            coords = v_res[:-1].reshape(N, 3)
            xs = coords[:, 0]
            ys = coords[:, 1]
            rs = coords[:, 2]
            
            # Validate
            valid = True
            for i in range(N):
                if xs[i] < rs[i] - 1e-5 or xs[i] + rs[i] > W + 1e-5:
                    valid = False
                    break
                if ys[i] < rs[i] - 1e-5 or ys[i] + rs[i] > H + 1e-5:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if d < rs[i] + rs[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_solution = coords.copy()
        
        except Exception as e:
            pass

    # If no valid solution found, return the seed solution
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
