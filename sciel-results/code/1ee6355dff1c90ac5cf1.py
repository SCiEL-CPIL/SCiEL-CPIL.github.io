"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    
    def is_valid(x, y, r, W, H):
        """Check if configuration is valid (disjoint, inside rectangle, perimeter <= 4)."""
        # Perimeter check
        if 2 * (W + H) > 4.0 + 1e-9:
            return False
        
        # Wall check
        if np.any(x < r) or np.any(W - x < r) or np.any(y < r) or np.any(H - y < r):
            return False
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist_sq = (r[i] + r[j])**2
                if dist_sq < min_dist_sq - 1e-9:
                    return False
        return True

    def objective_and_constraints(x, y, r, W, H):
        """
        Helper to compute objective and constraints for a specific W, H.
        We optimize x, y, r.
        Objective: -sum(r) (to maximize sum r)
        Constraints:
        1. x_i >= r_i
        2. W - x_i >= r_i
        3. y_i >= r_i
        4. H - y_i >= r_i
        5. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i < j
        """
        def obj(vars):
            r_vec = vars[2*N:]
            return -np.sum(r_vec)
        
        cons = []
        # Wall constraints (linear)
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            # W - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: W - v[i] - v[2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            # H - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[N+i] - v[2*N+i]})
            
        # Overlap constraints (non-linear)
        for i in range(N):
            for j in range(i + 1, N):
                def overlap_con(v, i=i, j=j):
                    dx = v[i] - v[j]
                    dy = v[N+i] - v[N+j]
                    dist_sq = dx*dx + dy*dy
                    r_sum = v[2*N+i] + v[2*N+j]
                    return dist_sq - r_sum*r_sum
                cons.append({'type': 'ineq', 'fun': overlap_con})
                
        return obj, cons

    def optimize_for_aspect(W_init, H_init, x_init, y_init, r_init):
        """
        Run SLSQP optimization for a fixed W, H.
        """
        x = x_init.copy()
        y = y_init.copy()
        r = r_init.copy()
        
        # Ensure initial validity
        # Shrink radii slightly if needed to ensure feasibility
        for i in range(N):
            r[i] = min(r[i], x[i], W_init - x[i], y[i], H_init - y[i])
        
        # Check overlaps and shrink
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 2*r[i]: # Assuming equal radii initially
                    r[i] = d/2 * 0.99
                    r[j] = d/2 * 0.99
        
        vars0 = np.concatenate([x, y, r])
        
        obj, cons = objective_and_constraints(x, y, r, W_init, H_init)
        
        # Bounds: x, y in [0, W/H], r in [0, min(W,H)/2]
        bounds = []
        for i in range(N):
            bounds.append((0, W_init))
        for i in range(N):
            bounds.append((0, H_init))
        for i in range(N):
            bounds.append((0, min(W_init, H_init)/2))
            
        try:
            res = minimize(obj, vars0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 100, 'ftol': 1e-6})
            
            if res.success or (res.status == 0):
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                # Verify validity
                if is_valid(x_opt, y_opt, r_opt, W_init, H_init):
                    sum_r = np.sum(r_opt)
                    if sum_r > best_sum_r:
                        best_sum_r = sum_r
                        best_config = np.column_stack((x_opt, y_opt, r_opt))
                        return best_sum_r
        except Exception as e:
            pass
            
        return -1.0

    # Main loop: Try different topologies and aspect ratios
    
    # 1. Grid Topologies
    # 2. Hex Topologies
    # 3. Random Topologies
    
    def generate_grid(rows, cols, W, H):
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                x[idx] = (j + 0.5) * W / cols
                y[idx] = (i + 0.5) * H / rows
                idx += 1
        # Fill remaining if any (should not happen for standard grids)
        while idx < N:
            x[idx] = W/2
            y[idx] = H/2
            idx += 1
        return x, y

    def generate_hex(W, H):
        # Approximate hex packing
        # Rows: 5, 4, 5, 4, 3 = 21
        counts = [5, 4, 5, 4, 3]
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        num_rows = len(counts)
        for i, count in enumerate(counts):
            for j in range(count):
                if idx >= N: break
                # Center in row
                x[idx] = (j + 0.5) * W / count
                y[idx] = (i + 0.5) * H / num_rows
                # Shift odd rows
                if i % 2 == 1:
                    x[idx] += W / (2 * count)
                idx += 1
        return x, y

    def generate_random(W, H):
        np.random.seed(42)
        x = np.random.uniform(0.1, W-0.1, N)
        y = np.random.uniform(0.1, H-0.1, N)
        return x, y

    topologies = [
        ("grid_4x5", 4, 5, "grid"),
        ("grid_5x4", 5, 4, "grid"),
        ("grid_3x7", 3, 7, "grid"),
        ("hex", None, None, "hex"),
        ("random", None, None, "random"),
    ]
    
    # Aspect ratio search
    # Perimeter 4 => W+H=2.
    # We will try a range of aspect ratios.
    # For each topology, we try a few aspect ratios.
    
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6] # W/H
    
    for name, rows, cols, type in topologies:
        if time.time() > deadline - 10:
            break
            
        for ar in aspect_ratios:
            if time.time() > deadline - 2:
                break
                
            W = 2.0 * ar / (ar + 1.0)
            H = 2.0 - W
            
            if W < 0.2 or H < 0.2:
                continue
                
            if type == "grid":
                x, y = generate_grid(rows, cols, W, H)
            elif type == "hex":
                x, y = generate_hex(W, H)
            else:
                x, y = generate_random(W, H)
                
            # Initial radius estimate
            # For grid: r ~ min(W/cols, H/rows)/2
            # For hex: similar
            # Just use a small safe radius
            r_init = np.full(N, 0.05)
            
            # Optimize
            optimize_for_aspect(W, H, x, y, r_init)
            
    # If no valid config found, return a trivial valid one
    if best_config is None:
        # Simple grid in 1x1 square (perimeter 4)
        W, H = 1.0, 1.0
        x, y = generate_grid(5, 5, W, H) # 25 circles, take 21
        r = np.full(N, 0.08)
        # Check validity
        if not is_valid(x, y, r, W, H):
            r = np.full(N, 0.05)
            if not is_valid(x, y, r, W, H):
                r = np.full(N, 0.01)
        best_config = np.column_stack((x, y, r))
        best_sum_r = np.sum(r)

    return best_config

# EVOLVE-BLOCK-END
