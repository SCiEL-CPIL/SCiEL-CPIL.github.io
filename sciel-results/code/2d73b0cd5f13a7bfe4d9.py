"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    
    best_score = -np.inf
    best_result = None
    
    # We will search over a range of initial aspect ratios.
    # By including 'w' as a variable in the optimization, the optimizer can adjust it.
    # Initial guesses for w (width) where h = 2 - w.
    # w=1.0 is square.
    initial_widths = [0.8, 0.9, 1.0, 1.1, 1.2, 0.7, 1.3]
    
    def unpack(vars_):
        # vars_ layout: x1..xN, y1..yN, r1..rN, w
        x = vars_[:N]
        y = vars_[N:2*N]
        r = vars_[2*N:3*N]
        w = vars_[3*N]
        h = 2.0 - w
        return x, y, r, w, h

    def objective(vars_):
        x, y, r, w, h = unpack(vars_)
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(r)

    def constraints_func(vars_):
        x, y, r, w, h = unpack(vars_)
        cons = []
        
        # 1. Boundary constraints
        # x - r >= 0
        cons.append(x - r)
        # w - x - r >= 0
        cons.append(w - x - r)
        # y - r >= 0
        cons.append(y - r)
        # h - y - r >= 0
        cons.append(h - y - r)
        
        # 2. Non-overlap constraints
        # dist(i,j) - ri - rj >= 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - r[i] - r[j])
                
        # 3. Positive radii
        cons.append(r)
        
        # 4. Width bounds (to avoid degenerate rectangles)
        # w >= 0.5, w <= 1.5 (since w+h=2)
        cons.append(w - 0.5)
        cons.append(1.5 - w)
        
        # Flatten list of arrays into a single array
        flat_cons = []
        for c in cons:
            if isinstance(c, np.ndarray):
                flat_cons.extend(c)
            else:
                flat_cons.append(c)
        return np.array(flat_cons)

    # Note: SLSQP handles inequality constraints g(x) >= 0
    # We need to define them as a single function returning an array.
    
    def make_slsqp_constraints():
        # SLSQP expects a list of dicts or a single function?
        # Actually, scipy.optimize.minimize with method='SLSQP' accepts:
        # constraints: list of dict or single dict?
        # Documentation says: "constraints : list of dict, optional"
        # Each dict has 'type' and 'fun'.
        # We can bundle all inequality constraints into one 'fun' returning an array.
        
        def all_ineq(vars_):
            x, y, r, w, h = unpack(vars_)
            # Boundaries
            c1 = x - r
            c2 = w - x - r
            c3 = y - r
            c4 = h - y - r
            
            # Overlaps
            overlaps = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    overlaps.append(dist - r[i] - r[j])
            c5 = np.array(overlaps)
            
            # Radii
            c6 = r
            
            # Width
            c7 = np.array([w - 0.5, 1.5 - w])
            
            return np.concatenate([c1, c2, c3, c4, c5, c6, c7])

        return [{'type': 'ineq', 'fun': all_ineq}]

    constraints = make_slsqp_constraints()

    for w_init in initial_widths:
        if time.time() > deadline - 10:
            break
            
        h_init = 2.0 - w_init
        
        # Generate initial positions
        # Estimate grid dimensions
        # Area ratio: N * pi * r^2 <= w * h
        # Approximate r ~ sqrt(w*h / (N * pi)) * packing_density_factor
        # But let's just use a grid based on aspect ratio.
        
        # Target number of columns
        # If w > h, more columns. If h > w, fewer columns.
        # ratio = w/h
        # cols ~ sqrt(N * ratio)
        ratio = w_init / h_init
        n_cols = int(np.sqrt(N * ratio))
        if n_cols < 1: n_cols = 1
        if n_cols > N: n_cols = N
        
        n_rows = int(np.ceil(N / n_cols))
        
        # Adjust if n_cols * n_rows < N (should not happen with ceil, but just in case)
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        # Fill grid
        idx = 0
        for r_idx in range(n_rows):
            for c_idx in range(n_cols):
                if idx >= N:
                    break
                # Center of cell
                x_c = (c_idx + 0.5) * w_init / n_cols
                y_c = (r_idx + 0.5) * h_init / n_rows
                
                # Add small jitter to break symmetry
                if idx > 0:
                    jitter = 0.05 * min(w_init/n_cols, h_init/n_rows)
                    x_c += np.random.uniform(-jitter, jitter)
                    y_c += np.random.uniform(-jitter, jitter)
                
                x0[idx] = x_c
                y0[idx] = y_c
                idx += 1
        
        # Calculate initial radii
        # Min distance to neighbors and walls
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x0[i]-x0[j])**2 + (y0[i]-y0[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, x0[i], w_init - x0[i], y0[i], h_init - y0[i])
            
        r0 = 0.4 * min_dist
        r0_arr = np.full(N, r0)
        
        # Assemble initial variable vector
        # [x1..xN, y1..yN, r1..rN, w]
        x_var = np.concatenate([x0, y0, r0_arr, [w_init]])
        
        # Run optimization
        try:
            result = minimize(
                objective, x_var, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success or result.fun < 0: # Check if we got a reasonable result
                x_res, y_res, r_res, w_res, h_res = unpack(result.x)
                
                # Validate
                valid = True
                if np.any(r_res < 0):
                    valid = False
                elif np.any(x_res < r_res - 1e-6) or np.any(x_res > w_res - r_res + 1e-6):
                    valid = False
                elif np.any(y_res < r_res - 1e-6) or np.any(y_res > h_res - r_res + 1e-6):
                    valid = False
                else:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_res[i]-x_res[j])**2 + (y_res[i]-y_res[j])**2)
                            if dist < r_res[i] + r_res[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_res, y_res, r_res]).copy()
                        
        except Exception:
            continue

    # Fallback
    if best_result is None:
        # Simple grid in square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        best_result = np.array(cells)
        
    return best_result
# EVOLVE-BLOCK-END
