import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial guess: Grid layout
    # We need to choose a grid size. 5x5 grid for 21 circles? Or 4x6?
    # Let's try a 5x5 grid, filling 21 spots.
    # Or better, a compact grid.
    # Let's use a square-ish grid. side = 5.
    side = int(np.ceil(np.sqrt(N)))
    
    # Initial positions in a unit square [0,1]x[0,1]
    # Then we will scale to fit the rectangle w x h.
    # Actually, it's better to initialize in the actual coordinate system.
    
    # Let's start with w=1.0, h=1.0 (square, perimeter 4).
    w_init = 1.0
    h_init = 1.0
    
    # Grid positions
    xs = []
    ys = []
    rs = []
    
    # Create a grid of points
    # We want to place N circles.
    # Let's use a 5x5 grid.
    grid_side = 5
    step = 1.0 / grid_side
    r_init = step / 2.0 * 0.9 # Slightly smaller to avoid initial overlap
    
    count = 0
    for i in range(grid_side):
        for j in range(grid_side):
            if count >= N:
                break
            x = (i + 0.5) * step
            y = (j + 0.5) * step
            xs.append(x)
            ys.append(y)
            rs.append(r_init)
            count += 1
        
    x0 = np.array(xs + ys + rs + [w_init])
    
    # Variables: x[0:N], y[N:2N], r[2N:3N], w[3N]
    # h = 2 - w
    
    def objective(v):
        if time.time() > deadline:
            return 1e10 # Signal to stop? No, minimize will just take a step.
            # Better to handle timeout outside or check inside constraints.
        
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        # If invalid dimensions, return large penalty
        if w <= 0.01 or h <= 0.01:
            return 1e10
            
        # Objective is negative sum of radii
        return -np.sum(r)

    def constraints(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        cons = []
        
        # Rectangle bounds
        # x_i >= r_i
        # x_i <= w - r_i
        # y_i >= r_i
        # y_i <= h - r_i
        
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[3*N]) - v[N+i] - v[2*N+i]})
            
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2})
                
        return cons

    # This creates too many constraints for SLSQP to be efficient if done naively with lambdas in a list.
    # It's better to define the constraint function to return an array.
    
    def constraint_func(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        
        c = []
        # Bounds
        c.extend(x - r)
        c.extend(w - x - r)
        c.extend(y - r)
        c.extend(h - y - r)
        
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum_sq = (r[i]+r[j])**2
                c.append(dist_sq - rad_sum_sq)
                
        return np.array(c)

    # SLSQP expects constraints as a list of dicts or a single dict with an array function.
    # Let's use the array function format.
    
    res = None
    best_score = -np.sum(rs)
    best_v = x0.copy()
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='SLSQP', 
                       constraints=[{'type': 'ineq', 'fun': constraint_func}],
                       bounds=[(1e-5, None)] * (3*N) + [(0.1, 1.9)],
                       options={'maxiter': 500, 'ftol': 1e-9})
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_v = res.x
        else:
            # Check if the result is valid even if not "success"
            score = -objective(res.x)
            if constraint_func(res.x).min() >= -1e-6: # Allow small tolerance
                if score > best_score:
                    best_score = score
                    best_v = res.x
            
    except Exception as e:
        pass

    # Try a few restarts if time permits
    restarts = 0
    while time.time() < deadline - 10 and restarts < 3:
        restarts += 1
        # Perturb the best solution
        v_pert = best_v.copy()
        v_pert[0:2*N] += np.random.normal(0, 0.01, 2*N) # Perturb positions
        # Ensure radii are positive
        v_pert[2*N:3*N] = np.maximum(v_pert[2*N:3*N], 1e-5)
        
        try:
            res2 = minimize(objective, v_pert, method='SLSQP', 
                           constraints=[{'type': 'ineq', 'fun': constraint_func}],
                           bounds=[(1e-5, None)] * (3*N) + [(0.1, 1.9)],
                           options={'maxiter': 300, 'ftol': 1e-9})
            
            if res2.success:
                score2 = -res2.fun
                if score2 > best_score:
                    best_score = score2
                    best_v = res2.x
            else:
                score2 = -objective(res2.x)
                if constraint_func(res2.x).min() >= -1e-6:
                    if score2 > best_score:
                        best_score = score2
                        best_v = res2.x
        except Exception:
            continue

    # Construct final output
    x_final = best_v[0:N]
    y_final = best_v[N:2*N]
    r_final = best_v[2*N:3*N]
    
    # Ensure strict validity (shrink radii slightly if needed to fix floating point errors)
    # Check constraints
    # If any constraint is violated by a tiny amount, shrink radii.
    
    # Simple validity check and fix
    # The evaluator checks for disjoint interiors and bounding box perimeter <= 4.
    # We should ensure the output is strictly valid.
    
    # Let's do a final polish or just return the best valid one.
    # To be safe, we can scale down the radii by a tiny factor if there are near-violations.
    
    # However, the objective maximizes sum of radii, so the optimizer pushes to the limit.
    # Floating point errors might make dist < r_i + r_j by 1e-16.
    # We can just return the coordinates. The evaluator usually has a tolerance or checks strictly.
    # "disjoint interiors" implies strict inequality.
    
    # Let's verify the best_v solution.
    # If it's invalid, we should fall back to the initial grid or a known valid configuration.
    
    # A simple valid configuration is the initial grid with slightly smaller radii.
    # Let's check if best_v is valid.
    def is_valid(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        if w <= 0 or h <= 0: return False
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9: return False
            if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9: return False
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum_sq = (r[i]+r[j])**2
                if dist_sq < rad_sum_sq - 1e-9: return False
        return True

    if not is_valid(best_v):
        # Fall back to a safe grid
        # Use the initial grid but with a safety margin
        w_safe = 1.0
        h_safe = 1.0
        grid_side = 5
        step = 1.0 / grid_side
        r_safe = step / 2.0 * 0.8 # 80% of max to ensure no overlap
        xs_safe = []
        ys_safe = []
        rs_safe = []
        count = 0
        for i in range(grid_side):
            for j in range(grid_side):
                if count >= N: break
                xs_safe.append((i + 0.5) * step)
                ys_safe.append((j + 0.5) * step)
                rs_safe.append(r_safe)
                count += 1
        out = np.array([xs_safe, ys_safe, rs_safe]).T
    else:
        out = np.array([x_final, y_final, r_final]).T

    return out
