import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_value = 0
    best_solution = None
    
    # Number of restarts to attempt
    # We will run as many as fit in the time budget
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # 1. Initialization
        # Try different initialization strategies
        if num_restarts % 2 == 0:
            # Grid initialization
            side = int(np.ceil(np.sqrt(N)))
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
                        rs.append(0.5 / side * 0.9) # Slightly smaller to start
        else:
            # Random initialization
            # Ensure they don't start overlapping heavily
            xs, ys, rs = [], [], []
            for _ in range(N):
                r = 0.05
                x = r + (1 - 2*r) * np.random.rand()
                y = r + (1 - 2*r) * np.random.rand()
                # Simple check for overlap with existing, retry if too close
                valid = True
                for k in range(len(xs)):
                    if (x - xs[k])**2 + (y - ys[k])**2 < (r + rs[k])**2:
                        valid = False
                        break
                if valid:
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
                else:
                    # Just place it anyway, optimizer will fix it, or try again
                    # To keep it simple, just place it.
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
        
        x0 = np.zeros(N * 3)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = rs[i]
        
        # 2. Define Objective and Constraints
        def objective(v):
            # Minimize negative sum of radii
            r = v[2::3]
            return -np.sum(r)
        
        def constraints(v):
            cons = []
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            
            # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}) # x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]}) # 1 - x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]}) # r_i >= 0
            
            # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                    })
            return cons
        
        # 3. Optimization
        try:
            # Bounds: x,y in [0,1], r in [0, 0.5]
            bounds = []
            for i in range(N):
                bounds.append((0, 1)) # x
                bounds.append((0, 1)) # y
                bounds.append((0, 0.5)) # r
            
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.success:
                val = -res.fun
                if val > best_value:
                    best_value = val
                    best_solution = res.x
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            break

    if best_solution is None:
        # Fallback to seed if optimization failed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format output: (x, y, radius)
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_solution[3*i]
        result[i, 1] = best_solution[3*i+1]
        result[i, 2] = best_solution[3*i+2]
        
    return result
# EVOLVE-BLOCK-END
