"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Initial guess: Hexagonal-like packing
    # We need a feasible starting point.
    # Let's create a grid and shrink radii to ensure feasibility.
    
    # 5 rows: 5, 6, 5, 6, 4 circles
    row_configs = [5, 6, 5, 6, 4]
    n_rows = len(row_configs)
    
    x_coords = []
    y_coords = []
    r_coords = []
    
    # Estimate max possible radius for this grid to be feasible
    # Horizontal spacing in row with 6 circles: 1/6
    # Vertical spacing: 5 rows in 1 unit.
    # Let's just start with small radii and let the optimizer grow them.
    # Initial radius: 0.05
    r_init = 0.05
    
    for i, n_in_row in enumerate(row_configs):
        y = (i + 0.5) / n_rows
        for j in range(n_in_row):
            x = (j + 0.5) / n_in_row
            x_coords.append(x)
            y_coords.append(y)
            r_coords.append(r_init)
            
    x0 = np.array(x_coords + y_coords + r_coords)
    
    # Ensure strict feasibility for start
    # Check overlaps and boundaries, shrink if needed
    # This simple check might be slow, but N=26 is small.
    # Actually, with r=0.05 and grid spacing >= 1/6 ~ 0.16, it should be safe.
    # 1/6 = 0.166. 2*r = 0.1. 0.166 > 0.1. Safe.
    # Boundary: x in [0.5/5, ...], min x ~ 0.1. r=0.05. 0.1-0.05=0.05 > 0. Safe.
    
    def objective(v):
        # v = [x1, y1, r1, ..., xN, yN, rN]
        # Maximize sum of radii => Minimize negative sum
        r = v[2::3]
        return -np.sum(r)
        
    def make_constraints(v):
        constraints = []
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Boundary constraints: g(v) >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}) # x_i - r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]}) # 1 - x_i - r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y_i - r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]}) # 1 - y_i - r_i
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]}) # r_i >= 0
            
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                def overlap_constraint(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    radius_sum_sq = (ri + rj)**2
                    return dist_sq - radius_sum_sq
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return constraints

    # COBYLA is good for non-smooth constraints and doesn't need gradients
    # However, defining constraints as a list of dicts is slow if done inside the loop.
    # It's better to define a single constraint function that returns an array of values.
    # scipy.optimize.minimize with COBYLA accepts a list of dicts OR a single dict with 'fun' returning an array.
    # Let's use the array format for efficiency.
    
    def constraint_array(v):
        vals = []
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Boundaries
        vals.append(x - r)
        vals.append(1 - x - r)
        vals.append(y - r)
        vals.append(1 - y - r)
        vals.append(r) # r >= 0
        
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                radius_sum_sq = (r[i] + r[j])**2
                vals.append(dist_sq - radius_sum_sq)
                
        return np.concatenate(vals)
    
    constraints = {'type': 'ineq', 'fun': constraint_array}
    
    # Bounds for x, y, r
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r (max possible is 0.5)
        
    # Run optimization
    # COBYLA can be slow. Let's set maxiter.
    # 120s limit.
    try:
        res = minimize(objective, x0, method='COBYLA', constraints=constraints, bounds=bounds, 
                       options={'maxiter': 2000, 'rhobeg': 0.01, 'catol': 1e-4})
        
        if res.success or res.fun < objective(x0):
            v_opt = res.x
        else:
            v_opt = x0
            
    except Exception:
        v_opt = x0
        
    # Extract result
    x_final = v_opt[0::3]
    y_final = v_opt[1::3]
    r_final = v_opt[2::3]
    
    # Ensure validity (COBYLA might not be perfectly strict)
    # Clip radii and positions to ensure they are inside
    for i in range(N):
        r_final[i] = min(r_final[i], x_final[i], 1 - x_final[i], y_final[i], 1 - y_final[i])
        r_final[i] = max(r_final[i], 0.0)
        
    # Final check and repair if invalid
    def is_valid(arr):
        n = len(arr)
        for i in range(n):
            x, y, r = arr[i]
            if x - r < -1e-5 or x + r > 1 + 1e-5 or y - r < -1e-5 or y + r > 1 + 1e-5:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((arr[i][0]-arr[j][0])**2 + (arr[i][1]-arr[j][1])**2)
                if dist < arr[i][2] + arr[j][2] - 1e-5:
                    return False
        return True
        
    result = np.column_stack((x_final, y_final, r_final))
    
    if not is_valid(result):
        # Fallback to a known valid configuration
        # Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
        result = np.array(cells)
        
    return result
# EVOLVE-BLOCK-END
