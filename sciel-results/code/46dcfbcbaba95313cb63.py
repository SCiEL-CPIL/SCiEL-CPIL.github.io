import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_circles = None
    
    # Objective: Maximize sum of radii -> Minimize -sum(r)
    # Variables: x1, y1, r1, ..., xN, yN, rN, W
    # H is derived as 2 - W
    
    def objective(v):
        r = v[2::3][:N]
        return -np.sum(r)
        
    def make_constraints():
        """
        Returns a list of constraint dictionaries for SLSQP.
        """
        cons = []
        
        # 1. Boundary Constraints
        # x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[0::3][:N] - v[2::3][:N]
        })
        # W - x - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[-1] - v[0::3][:N] - v[2::3][:N]
        })
        # y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: v[1::3][:N] - v[2::3][:N]
        })
        # H - y - r >= 0  =>  (2 - W) - y - r >= 0
        cons.append({
            'type': 'ineq',
            'fun': lambda v: (2.0 - v[-1]) - v[1::3][:N] - v[2::3][:N]
        })
        
        # 2. Non-overlap Constraints
        # dist(i,j) - r_i - r_j >= 0
        def non_overlap(v):
            x = v[0::3][:N]
            y = v[1::3][:N]
            r = v[2::3][:N]
            cons_val = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons_val.append(dist - r[i] - r[j])
            return np.array(cons_val)
        
        cons.append({
            'type': 'ineq',
            'fun': non_overlap
        })
        
        return cons

    constraints = make_constraints()
    
    # Bounds for SLSQP
    # x, y in [0, 2] (since W+H=2, max dim is 2)
    # r in [0, 1]
    # W in [0.5, 1.5] (avoid degenerate)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
        bounds.append((0.0, 2.0)) # y
        bounds.append((0.0, 1.0)) # r
    bounds.append((0.5, 1.5)) # W
    
    np.random.seed(42)
    
    # Number of restarts to try
    # We have ~115s. SLSQP might take 1-5s per run. Let's try 10-20 restarts.
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Initialization: Grid layout in a square (W=1, H=1)
        # 5x5 grid = 25 spots. We need 21.
        # We will use the first 21 spots or randomly select 21.
        # To add diversity, we can jitter the positions or select different subsets.
        
        W0 = 1.0
        H0 = 1.0
        
        x0 = np.zeros(3*N + 1)
        
        # Generate 25 grid points
        grid_points = []
        for i in range(5):
            for j in range(5):
                cx = (j + 0.5) * W0 / 5
                cy = (i + 0.5) * H0 / 5
                grid_points.append([cx, cy])
                
        # Randomly select 21 points for this restart
        # Use a different seed for each restart to get different subsets/jitters
        np.random.seed(42 + restart)
        selected_indices = np.random.choice(25, N, replace=False)
        
        for k, idx in enumerate(selected_indices):
            cx, cy = grid_points[idx]
            
            # Add jitter to escape local minima
            jitter = 0.05 * (restart / 5.0 + 0.1) # Increase jitter slightly with restart
            cx += np.random.uniform(-jitter, jitter)
            cy += np.random.uniform(-jitter, jitter)
            
            # Clamp to inside
            cx = np.clip(cx, 0.05, W0 - 0.05)
            cy = np.clip(cy, 0.05, H0 - 0.05)
            
            x0[3*k] = cx
            x0[3*k+1] = cy
            x0[3*k+2] = 0.08 # Initial radius
            
        x0[-1] = W0 # Initial W
        
        # Run SLSQP
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            if res.success or (res.fun < -1e-5):
                x_res = res.x[0::3][:N]
                y_res = res.x[1::3][:N]
                r_res = res.x[2::3][:N]
                W_res = res.x[-1]
                H_res = 2.0 - W_res
                
                # Validation
                valid = True
                if np.any(r_res < -1e-6):
                    valid = False
                else:
                    r_res = np.maximum(r_res, 0)
                    
                    # Check boundaries
                    if np.any(x_res < r_res - 1e-5) or np.any(x_res > W_res - r_res + 1e-5):
                        valid = False
                    if np.any(y_res < r_res - 1e-5) or np.any(y_res > H_res - r_res + 1e-5):
                        valid = False
                    
                    # Check overlaps
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = x_res[i] - x_res[j]
                                dy = y_res[i] - y_res[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < r_res[i] + r_res[j] - 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_circles = np.column_stack((x_res, y_res, r_res)).copy()
                        
        except Exception:
            pass

    # Fallback if nothing found
    if best_circles is None:
        # Simple grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([
                        (i + 0.5) / side,
                        (j + 0.5) / side,
                        0.5 / side - 0.01
                    ])
        best_circles = np.array(cells[:N])
        
    return best_circles
