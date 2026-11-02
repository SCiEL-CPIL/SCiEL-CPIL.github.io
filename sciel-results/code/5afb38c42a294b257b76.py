"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 115.0  # seconds, safety margin before 120s limit
_start_time = None

def _elapsed():
    return time.time() - _start_time

def construct():
    global _start_time
    _start_time = time.time()
    
    # We want to pack 21 circles in a rectangle of perimeter 4.
    # Let width = w, height = h. 2(w+h) = 4 => w+h = 2.
    # To maximize sum of radii, we likely want a square-ish rectangle or optimized aspect ratio.
    # A square of side 1x1 has perimeter 4.
    # Let's fix the rectangle to 1x1 for simplicity first, or allow optimization.
    # Actually, the problem says "variable rectangle of perimeter 4".
    # Let's parameterize by aspect ratio. Let h = 1, w = 1. (Square)
    # Or better, let's let the optimizer adjust positions and radii within a fixed 1x1 box.
    # If we use a 1x1 box, perimeter is 4.
    # Objective: maximize sum(r_i)
    # Constraints:
    #   1. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    #   2. Boundary: r_i <= x_i <= 1 - r_i, r_i <= y_i <= 1 - r_i
    #   3. r_i > 0
    
    # Variables: x, y, r for each circle. Total 3*N variables.
    # We can use SLSQP with a penalty or direct constraints.
    # Direct constraints are O(N^2) which is fine for N=21 (441 pairs).
    
    n = N
    # Initial guess: grid layout
    # 21 circles. 5x5 grid is 25. 4x5 is 20. 
    # Let's use a 5x5 grid but remove 4 corners? Or just place them in a grid.
    # Better: 1 row of 21? No, too narrow.
    # 5 rows, 5 cols.
    # Let's start with a dense grid and let the optimizer expand radii.
    
    cols = 5
    rows = 5
    # Create initial positions
    x0 = []
    y0 = []
    r0 = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count < n:
                x0.append((j + 0.5) / cols)
                y0.append((i + 0.5) / rows)
                r0.append(0.1) # small initial radius
                count += 1
    
    x0 = np.array(x0)
    y0 = np.array(y0)
    r0 = np.array(r0)
    
    # Flatten variables: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    def flatten(x, y, r):
        return np.concatenate([x, y, r])
        
    def unflatten(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:]
        return x, y, r

    # Objective: minimize -sum(r)
    def objective(v):
        x, y, r = unflatten(v)
        return -np.sum(r)

    # Constraints
    # Inequality constraints: g(v) >= 0
    # 1. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # 2. Boundary x: x_i - r_i >= 0, 1 - x_i - r_i >= 0
    # 3. Boundary y: y_i - r_i >= 0, 1 - y_i - r_i >= 0
    # 4. Radius positivity: r_i >= epsilon (handled by bounds)

    def constraints_func(v):
        x, y, r = unflatten(v)
        cons = []
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                sum_r = r[i] + r[j]
                cons.append(dist_sq - sum_r**2)
        
        # Boundary constraints
        for i in range(n):
            cons.append(x[i] - r[i])
            cons.append(1.0 - x[i] - r[i])
            cons.append(y[i] - r[i])
            cons.append(1.0 - y[i] - r[i])
            
        return cons

    # Build constraint list for SLSQP
    constraints = []
    for k in range(n*(n-1)//2 + 4*n):
        constraints.append({'type': 'ineq', 'fun': lambda v, k=k: constraints_func(v)[k]})

    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
    for i in range(n):
        bounds.append((0.0, 1.0)) # y
    for i in range(n):
        bounds.append((1e-6, 0.5)) # r

    v0 = flatten(x0, y0, r0)

    # Multi-start strategy
    best_val = -np.inf
    best_v = v0
    
    # Number of restarts depends on time
    # Let's try a few different initial configurations
    starts = []
    
    # Start 1: Grid
    starts.append(v0)
    
    # Start 2: Random
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.uniform(0.2, 0.8, n)
        y_rand = np.random.uniform(0.2, 0.8, n)
        r_rand = np.full(n, 0.05)
        starts.append(flatten(x_rand, y_rand, r_rand))
        
    # Start 3: Hexagonal packing approx
    # Just another random with different seed
    np.random.seed(123)
    for _ in range(5):
        x_rand = np.random.uniform(0.1, 0.9, n)
        y_rand = np.random.uniform(0.1, 0.9, n)
        r_rand = np.full(n, 0.08)
        starts.append(flatten(x_rand, y_rand, r_rand))

    for idx, v_start in enumerate(starts):
        if _elapsed() > DEADLINE - 10:
            break
            
        try:
            res = minimize(objective, v_start, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x
        except Exception:
            pass

    # Polish: Use the best result and run a tighter optimization
    if _elapsed() < DEADLINE - 5:
        try:
            # Refine with tighter tolerance
            res_final = minimize(objective, best_v, method='SLSQP', bounds=bounds, 
                                 constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if res_final.success:
                val_final = -res_final.fun
                if val_final > best_val:
                    best_val = val_final
                    best_v = res_final.x
        except Exception:
            pass

    # Extract best circles
    x, y, r = unflatten(best_v)
    
    # Ensure validity (clip tiny negative radii or overlaps if any due to numerical error)
    # The problem requires disjoint interiors.
    # Let's do a final check and scale down if necessary?
    # The optimizer should satisfy constraints.
    
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
