import numpy as np
import scipy.optimize as opt
import time

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_penalty_objective(params):
        """
        Objective for Differential Evolution.
        Minimizes -sum(r) + penalties.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure r is positive for penalty calculation stability, though DE bounds handle it
        r = np.maximum(r, 0)
        
        # Objective: -sum(r)
        obj = -np.sum(r)
        
        # Overlap Penalty
        # Vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        overlap = np.maximum(0, r_sum - d)
        
        # Sum over i < j
        iu = np.triu_indices(N, k=1)
        penalty_overlap = np.sum(overlap[iu] ** 2)
        
        # Boundary Penalty
        # x >= r, x <= 1-r
        # y >= r, y <= 1-r
        pen_bound = 0.0
        pen_bound += np.sum(np.maximum(0, r - x) ** 2)
        pen_bound += np.sum(np.maximum(0, r - (1 - x)) ** 2)
        pen_bound += np.sum(np.maximum(0, r - y) ** 2)
        pen_bound += np.sum(np.maximum(0, r - (1 - y)) ** 2)
        
        # Scale penalties to be significant
        # If obj is around -2.0, penalties should be large if invalid
        return obj + 100.0 * penalty_overlap + 100.0 * pen_bound

    def get_slsqp_objective(params):
        """
        Objective for SLSQP.
        Simply -sum(r). Constraints handle validity.
        """
        r = params[2*N:3*N]
        return -np.sum(r)

    def get_slsqp_constraints():
        """
        Returns list of constraint dictionaries for SLSQP.
        Constraint: fun(x) >= 0
        """
        cons = []
        
        # Overlap constraints: d_ij - r_i - r_j >= 0
        # We can vectorize the calculation inside the fun, but SLSQP calls fun(x) for each constraint.
        # To be faster, we can define a single function that returns an array of values, 
        # but scipy.optimize.minimize with method='SLSQP' expects a list of dicts or a single dict with an array fun.
        # Let's use a single dict with an array fun for all overlaps and bounds to reduce overhead?
        # No, SLSQP handles arrays well if we provide 'ineq' as an array.
        
        # Actually, providing a list of 325+104 constraints is slow in Python loop overhead.
        # It is better to define one constraint function that returns an array of all constraint values.
        
        def constraint_fun(params):
            x = params[:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            # Overlaps
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            r_sum = r[:, None] + r[None, :]
            
            # We need d_ij - r_i - r_j >= 0 for i != j
            # Create matrix C where C_ij = d_ij - r_i - r_j
            C = d - r_sum
            # Set diagonal to 0 (or large positive) as we don't care about i=j
            np.fill_diagonal(C, 1.0)
            
            # Extract upper triangle
            iu = np.triu_indices(N, k=1)
            overlap_cons = C[iu]
            
            # Boundaries
            # x - r >= 0
            c_x_min = x - r
            # 1 - x - r >= 0
            c_x_max = 1 - x - r
            # y - r >= 0
            c_y_min = y - r
            # 1 - y - r >= 0
            c_y_max = 1 - y - r
            
            # Combine all into one array
            all_cons = np.concatenate([overlap_cons, c_x_min, c_x_max, c_y_min, c_y_max])
            return all_cons
            
        return [{'type': 'ineq', 'fun': constraint_fun}]

    def is_valid(params, tol=1e-5):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
            
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        # d_ij >= r_i + r_j
        # Check if any d_ij < r_i + r_j - tol
        iu = np.triu_indices(N, k=1)
        if np.any(d[iu] < r_sum[iu] - tol):
            return False
        return True

    def get_initial_guess():
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N:
                    break
                pts.append((xs[i], ys[j]))
            if len(pts) >= N:
                break
        
        while len(pts) < N:
            pts.append((0.5, 0.5))
            
        pts = np.array(pts[:N])
        
        r_init = 0.45 / side
        
        x = pts[:, 0]
        y = pts[:, 1]
        r = np.full(N, r_init)
        
        # Add small noise
        x += np.random.normal(0, 0.01, N)
        y += np.random.normal(0, 0.01, N)
        r += np.random.normal(0, 0.005, N)
        
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
        r = np.clip(r, 0.01, 0.2)
        
        return np.concatenate([x, y, r])

    best_params = None
    best_val = -np.inf
    
    # Phase 1: Differential Evolution
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    # DE can be slow. Use a reasonable popsize and maxiter.
    # For 78 dims, popsize=10 means 780 individuals.
    # Let's try maxiter=50 to see how fast it is.
    
    try:
        de_res = opt.differential_evolution(
            get_penalty_objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=False, # We will do our own SLSQP polish
            init='sobol' # Better initialization
        )
        
        if time.time() > deadline - 30:
             # If DE took too long, skip to fallback or just use DE result if valid
             pass
        else:
            # Check DE result
            if is_valid(de_res.x):
                val = np.sum(de_res.x[2*N:3*N])
                if val > best_val:
                    best_val = val
                    best_params = de_res.x.copy()
            
            # Also check a few other points from DE history?
            # DE doesn't expose history easily in older scipy, but we can just use the best.
            
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE didn't find a valid solution, use initial guess
    if best_params is None:
        best_params = get_initial_guess()
        # Check if initial guess is valid (it might not be due to noise)
        if not is_valid(best_params):
            # Shrink radii to make valid
            r = best_params[2*N:3*N]
            for scale in [0.9, 0.8, 0.7, 0.5]:
                r_new = r * scale
                best_params[2*N:3*N] = r_new
                if is_valid(best_params):
                    break

    # Phase 2: SLSQP Polish
    # Use the best_params from DE (or fallback) as starting point for SLSQP
    
    # We can try SLSQP on the best_params
    # And maybe on a few other random starts if time permits?
    # SLSQP is sensitive to start. DE provides a good start.
    
    start_slsqp_time = time.time()
    
    try:
        # SLSQP with hard constraints
        # Objective: -sum(r)
        # Constraints: ineq array
        
        cons = get_slsqp_constraints()
        
        # Bounds for SLSQP
        # x, y in [0, 1], r in [0, 0.5]
        slsqp_bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        # Run SLSQP
        # maxiter needs to be high enough to converge.
        res_slsqp = opt.minimize(
            get_slsqp_objective,
            best_params,
            method='SLSQP',
            bounds=slsqp_bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res_slsqp.success and is_valid(res_slsqp.x):
            val = np.sum(res_slsqp.x[2*N:3*N])
            if val > best_val:
                best_val = val
                best_params = res_slsqp.x.copy()
        elif is_valid(res_slsqp.x):
            # Even if not "success" (e.g. maxiter reached), if it's valid, take it
            val = np.sum(res_slsqp.x[2*N:3*N])
            if val > best_val:
                best_val = val
                best_params = res_slsqp.x.copy()

    except Exception as e:
        print(f"SLSQP failed: {e}")

    # If time remains, try SLSQP from a few other random starts?
    # DE might have found a local minimum.
    # But DE is global, so the best DE result is likely the best basin.
    # However, SLSQP from a *slightly perturbed* best_params might escape a tiny local min?
    # Unlikely to help much if SLSQP converged.
    
    # Let's try one more SLSQP from a different initial guess if time > 20s remaining
    if time.time() < deadline - 20:
        # Try a second SLSQP from a random start
        x0_2 = get_initial_guess()
        if is_valid(x0_2):
             try:
                 res_slsqp_2 = opt.minimize(
                    get_slsqp_objective,
                    x0_2,
                    method='SLSQP',
                    bounds=slsqp_bounds,
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                 if is_valid(res_slsqp_2.x):
                    val = np.sum(res_slsqp_2.x[2*N:3*N])
                    if val > best_val:
                        best_val = val
                        best_params = res_slsqp_2.x.copy()
             except:
                 pass

    # Final check and return
    if best_params is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        arr = np.array(cells[:N])
        return np.column_stack([arr[:,0], arr[:,1], arr[:,2]])

    # Reshape to (N, 3)
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
