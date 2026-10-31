"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # We need to pack 21 circles in a rectangle of perimeter 4.
    # Let the rectangle have width w and height h.
    # Perimeter constraint: 2(w + h) = 4  =>  w + h = 2.
    # We want to maximize the sum of radii: R = sum(r_i).
    # Constraints:
    # 1. Circles are inside the rectangle:
    #    r_i <= x_i <= w - r_i
    #    r_i <= y_i <= h - r_i
    # 2. Circles do not overlap:
    #    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 3. w + h = 2, w > 0, h > 0.
    
    # We can fix the aspect ratio or optimize over it.
    # Let's parameterize the rectangle by its aspect ratio or just fix w and h.
    # Since the problem is symmetric in w and h, we can assume w >= h.
    # Let w = 1 + delta, h = 1 - delta, where delta in [0, 1).
    # Actually, let's just use w and h as variables, with the constraint w + h = 2.
    # To simplify, we can fix one dimension and optimize the other? No, both matter.
    # Let's use a parameter t such that w = 2 * t, h = 2 * (1-t) is wrong because w+h=2.
    # Let w = 1 + a, h = 1 - a. Then w+h=2. a in [-1, 1].
    # Since the problem is symmetric, let's assume a >= 0, so w >= 1, h <= 1.
    # Actually, the optimal rectangle for circle packing is often close to square.
    # Let's try to optimize the positions and radii for a fixed aspect ratio first,
    # then optimize the aspect ratio.
    
    # Alternatively, we can treat w and h as variables too.
    # Variables: x_i, y_i, r_i for i=1..N, and w, h.
    # Total variables: 3*N + 2 = 65.
    # Objective: Maximize sum(r_i)
    # Constraints:
    #   w + h = 2 (equality)
    #   w > 0, h > 0
    #   r_i <= x_i <= w - r_i
    #   r_i <= y_i <= h - r_i
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    # This is a non-convex problem. We can use a local optimizer like SLSQP or trust-constr.
    # We need a good initial guess.
    
    # Let's start with a grid arrangement.
    # For 21 circles, a 5x4 grid + 1 extra? Or 6x4?
    # Let's try a 5x5 grid, but only use 21 cells.
    # Or better, use a hexagonal packing initial guess.
    
    # Let's define the variables as a flat vector:
    # [x_1, y_1, r_1, ..., x_N, y_N, r_N, w, h]
    
    # Initial guess:
    # Let's assume a square-ish rectangle, say w=1.2, h=0.8 (sum=2).
    # Or w=1.0, h=1.0.
    # Let's try w=1.0, h=1.0 first.
    
    def initial_guess(w, h):
        # Create a grid of points
        # We want to place 21 points in a w x h rectangle.
        # Let's use a grid of 5x5, but only take the first 21 points.
        # Or better, use a hexagonal lattice.
        
        # Simple grid:
        cols = 5
        rows = 5
        # We only need 21 points.
        # Let's place them in a 5x5 grid, centered in the rectangle.
        xs = []
        ys = []
        rs = []
        
        # Grid spacing
        dx = w / cols
        dy = h / rows
        
        # Place points at grid centers
        for i in range(cols):
            for j in range(rows):
                if len(xs) >= N:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                xs.append(x)
                ys.append(y)
        
        # Initial radius: half the minimum distance between any two points, or to the wall.
        # Let's set a small initial radius.
        r0 = 0.05
        rs = [r0] * N
        
        return np.array(xs + ys + rs + [w, h])
    
    def objective(v):
        # v = [x_1, y_1, r_1, ..., x_N, y_N, r_N, w, h]
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        w = v[3*N]
        h = v[3*N+1]
        
        # We want to maximize sum(rs), so minimize -sum(rs)
        return -np.sum(rs)
    
    def constraints(v):
        xs = v[0:N]
        ys = v[N:2*N]
        rs = v[2*N:3*N]
        w = v[3*N]
        h = v[3*N+1]
        
        cons = []
        
        # Perimeter constraint: w + h = 2
        cons.append({'type': 'eq', 'fun': lambda v: v[3*N] + v[3*N+1] - 2})
        
        # w > 0, h > 0
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*N] - 0.1})
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*N+1] - 0.1})
        
        # Wall constraints:
        # r_i <= x_i <= w - r_i  =>  x_i - r_i >= 0, w - x_i - r_i >= 0
        # r_i <= y_i <= h - r_i  =>  y_i - r_i >= 0, h - y_i - r_i >= 0
        
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            # h - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
            
        # Circle-circle non-overlap:
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # This is not a linear constraint, but we can write it as:
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        # This is a non-convex constraint. SLSQP can handle it, but it might be tricky.
        # Alternatively, we can use a penalty method.
        
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
                
        return cons
    
    # Let's try a few initial aspect ratios.
    best_val = -np.inf
    best_v = None
    
    # Try different aspect ratios
    for w_ratio in [0.8, 0.9, 1.0, 1.1, 1.2]:
        w = 1.0 + (w_ratio - 1.0) * 0.5  # Just a guess
        # Actually, let's just use w = w_ratio * 2 / (w_ratio + 1) * something?
        # Simpler: let w = w_ratio, h = 2 - w_ratio.
        # Ensure w, h > 0.1
        if w < 0.1 or w > 1.9:
            continue
        h = 2 - w
        
        v0 = initial_guess(w, h)
        
        # Check if time is running out
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(objective, v0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-6})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x
        except Exception as e:
            pass
            
        if time.time() > deadline - 10:
            break
            
    if best_v is None:
        # Fallback to a simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract the result
    xs = best_v[0:N]
    ys = best_v[N:2*N]
    rs = best_v[2*N:3*N]
    
    # Ensure feasibility (small numerical errors might violate constraints)
    # We can shrink the radii slightly if needed.
    # For now, return the result as is.
    
    result = np.column_stack((xs, ys, rs))
    
    # Verify perimeter
    w = best_v[3*N]
    h = best_v[3*N+1]
    perimeter = 2 * (w + h)
    # If perimeter is slightly over 4, scale down the whole configuration.
    if perimeter > 4:
        scale = 4 / perimeter
        result[:, 0] *= scale
        result[:, 1] *= scale
        result[:, 2] *= scale
        
    return result
# EVOLVE-BLOCK-END
