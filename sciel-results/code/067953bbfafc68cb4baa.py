import numpy as np
import time
from scipy.optimize import minimize

N = 32
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_solution = None
    
    # Helper to check time
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)

    def objective_and_constraints(z):
        """
        z: array of shape (3*N)
        Objective: minimize -sum(r)
        Returns: (-sum_r, constraints_list)
        """
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Objective
        f = -np.sum(r)
        
        # Constraints
        cons = []
        
        # Boundary constraints
        # x >= r
        cons.append({'type': 'ineq', 'fun': lambda z, x=x, r=r: x - r}) # This lambda capture is wrong for batch
        # Better to define vectorized constraint functions or list of scalar constraints
        # SLSQP supports vectorized constraints if 'fun' returns array? No, usually scalar or array of constraints.
        # Let's use a single constraint function returning an array of values >= 0.
        
        # Actually, SLSQP 'cons' can be a list of dicts, or a single dict with 'fun' returning array.
        # If 'fun' returns an array, each element is a constraint >= 0.
        
        # Let's build the array of constraint values
        # 1. x_i - r_i >= 0
        # 2. 1 - x_i - r_i >= 0
        # 3. y_i - r_i >= 0
        # 4. 1 - y_i - r_i >= 0
        # 5. (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        # We will compute these outside the lambda for efficiency if possible, 
        # but SLSQP calls 'fun' repeatedly.
        
        return f, x, y, r

    def calc_constraints(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        c_bounds = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        c_pairwise = []
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum = r[i] + r[j]
                c_pairwise.append(dist_sq - rad_sum**2)
        
        return np.concatenate([c_bounds, c_pairwise])

    def obj_func(z):
        r = z[2::3]
        return -np.sum(r)

    def cons_func(z):
        return calc_constraints(z)

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    # Reshape to match z structure: [x1, y1, r1, x2, y2, r2, ...]
    # My z structure is [x1, x2, ..., xN, y1, ..., yN, r1, ..., rN]
    # Let's stick to interleaved: z = [x1, y1, r1, x2, y2, r2, ...]
    
    # Re-indexing for interleaved
    def obj_func_int(z):
        r = z[2::3]
        return -np.sum(r)
        
    def cons_func_int(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        c_bounds = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        c_pairwise = []
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum = r[i] + r[j]
                c_pairwise.append(dist_sq - rad_sum**2)
        
        return np.concatenate([c_bounds, c_pairwise])

    bounds_int = []
    for _ in range(N):
        bounds_int.append((0, 1)) # x
        bounds_int.append((0, 1)) # y
        bounds_int.append((0, 0.5)) # r

    constraints = [
        {'type': 'ineq', 'fun': cons_func_int}
    ]

    def run_optimizer(z0):
        res = minimize(
            obj_func_int, 
            z0, 
            method='SLSQP', 
            bounds=bounds_int, 
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        return res

    # Initialization Strategies
    
    # 1. Hexagonal Initialization
    def init_hex():
        # Generate hex lattice points
        # Spacing s
        # We want ~32 points. sqrt(32) ~ 5.6. 
        # Hex packing density is pi/sqrt(12) ~ 0.9069.
        # Area per circle ~ 1/32. r ~ sqrt(1/(32*pi)) ~ 0.099.
        # Distance ~ 2r ~ 0.2.
        
        # Let's try to fit a hex grid.
        # Rows of circles.
        # Row 0: y = r
        # Row 1: y = r + sqrt(3)*r
        # etc.
        
        # Simpler: Generate a grid of points, then optimize.
        # Use a slightly perturbed grid to break symmetry.
        side = int(np.ceil(np.sqrt(N)))
        # For N=32, side=6 gives 36 cells. We take 32.
        # Or use a rectangular grid 4x8?
        
        # Let's use a hex-like arrangement manually.
        # 4 rows of 8? 
        # Or 6x6 grid, remove 4 corners?
        
        # Let's stick to a simple grid for init, but perturb it.
        # The optimizer will move them.
        
        # Better Init: Random
        return None

    def init_random():
        np.random.seed(42) # Fixed seed for reproducibility in this run, but we can vary it
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        # 6x6 grid
        xs, ys = np.meshgrid(np.linspace(0.5/6, 1-0.5/6, 6), np.linspace(0.5/6, 1-0.5/6, 6))
        x = xs.flatten()
        y = ys.flatten()
        # Take first 32
        x = x[:N]
        y = y[:N]
        r = np.full(N, 0.08)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    # Run multiple restarts
    initializations = [
        init_grid,
        init_random,
    ]
    
    # Add perturbed grid
    def init_perturbed_grid():
        z = init_grid()
        noise = np.random.uniform(-0.05, 0.05, 3*N)
        # Keep within bounds roughly
        z += noise
        z = np.clip(z, 0, 1)
        # Radii should be positive
        z[2::3] = np.maximum(z[2::3], 0.01)
        return z
    
    initializations.append(init_perturbed_grid)
    
    # Hex initialization attempt
    def init_hexagonal():
        # Approximate hex packing
        # 1 row of 8, 1 row of 7, 1 row of 8, 1 row of 7, 1 row of 8? 
        # Total 38. Too many.
        # 4 rows of 8 = 32.
        # Spacing in x: 1/8 = 0.125
        # Spacing in y: 0.125 * sqrt(3) ~ 0.216
        # Total height: 4 * 0.216 = 0.864. Fits in 1.0.
        
        x = []
        y = []
        for i in range(4):
            offset = 0 if i % 2 == 0 else 0.0625
            for j in range(8):
                x.append(offset + j * 0.125 + 0.0625)
                y.append(0.15 + i * 0.216)
        
        x = np.array(x)
        y = np.array(y)
        r = np.full(N, 0.06)
        
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    initializations.append(init_hexagonal)
    
    for i, init_func in enumerate(initializations):
        if time_remaining() < 10:
            break
            
        z0 = init_func()
        # Add small random noise to break symmetry if deterministic
        if i > 0:
             z0 += np.random.uniform(-0.01, 0.01, 3*N)
             z0 = np.clip(z0, 0, 1)
             z0[2::3] = np.maximum(z0[2::3], 0.01)

        try:
            res = run_optimizer(z0)
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_solution = res.x
        except Exception as e:
            pass

    # If we have time, do a polish on the best solution
    if best_solution is not None and time_remaining() > 5:
        try:
            # Tighter tolerance
            res = minimize(
                obj_func_int, 
                best_solution, 
                method='SLSQP', 
                bounds=bounds_int, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_solution = res.x
        except:
            pass

    # Format output
    if best_solution is None:
        # Fallback to grid
        z0 = init_grid()
        best_solution = z0
        
    x = best_solution[0::3]
    y = best_solution[1::3]
    r = best_solution[2::3]
    
    # Ensure validity (clip to bounds, enforce non-overlap if slightly violated due to numerical error)
    # The optimizer should have handled this, but let's be safe.
    # If constraints are violated, shrink radii slightly?
    # For now, return as is.
    
    result = np.column_stack((x, y, r))
    return result
