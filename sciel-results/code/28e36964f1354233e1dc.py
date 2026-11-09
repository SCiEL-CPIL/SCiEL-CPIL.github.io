import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            # Clip to ensure we stay in bounds, though func also clips
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Structured Initialization
        # We want to cover the space [-4, 4]^5 reasonably well.
        # 1. Center
        init_points = [np.zeros(dim)]
        
        # 2. Axis-aligned points (2 per axis, +/- 2.0)
        axis_step = 2.0
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * axis_step
                init_points.append(x)
        
        # 3. Diagonal points in first 2 dims (4 points)
        for sx in [1, -1]:
            for sy in [1, -1]:
                x = np.zeros(dim)
                x[0] = sx * axis_step
                x[1] = sy * axis_step
                init_points.append(x)
                
        # 4. Random points to fill gaps
        # We have used 1 + 2*5 + 4 = 16 points.
        # Budget is 500. We can afford more. Let's add 8 random points.
        for _ in range(8):
            x = rng.uniform(-4.5, 4.5, dim)
            init_points.append(x)
            
        # Evaluate all init points
        for x in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = call_func(x)
            
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead
        start_x = best_x.copy()
        
        # Initialize simplex
        simplex = [start_x.copy()]
        simplex_f = [best_f]
        
        # Step size for simplex initialization
        step_size = 0.8
        for i in range(dim):
            v = start_x.copy()
            v[i] += step_size
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)
            
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        max_iter = 150 # Limit iterations to leave budget for polish
        tol = 1e-10
        no_improve_count = 0
        best_f_nm = best_f
        
        for iteration in range(max_iter):
            if calls_made >= budget - 50 or time.time() > deadline - 10: # Leave 50 for polish
                break
            
            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < tol:
                break
            
            # Track no-improvement
            if simplex_f[0] < best_f_nm - 1e-15:
                no_improve_count = 0
                best_f_nm = simplex_f[0]
            else:
                no_improve_count += 1
                
            # Restart logic: if stuck, try a random point near best
            if no_improve_count >= 20 and calls_made < budget - 50:
                # Generate a new point near current best
                x_new = simplex[0] + rng.normal(0, 0.5, dim)
                x_new = np.clip(x_new, -4.5, 4.5)
                f_new = call_func(x_new)
                if f_new < simplex_f[0]:
                    simplex = [x_new.copy()]
                    simplex_f = [f_new]
                    for i in range(dim):
                        v = x_new.copy()
                        v[i] += 0.5
                        fv = call_func(v)
                        simplex.append(v.copy())
                        simplex_f.append(fv)
                    no_improve_count = 0
                    best_f_nm = f_new
                    continue
                else:
                    # If restart failed, reduce step size or break?
                    # Let's just continue, but maybe shrink simplex more aggressively
                    pass

            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)
            
            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                f_con = call_func(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = call_func(simplex[i])
                        
        # Phase 3: Axis-Aligned Line Search Polish
        # Use remaining budget to refine each dimension independently.
        # This is effective for the final convergence on rotated spheres.
        
        remaining_budget = budget - calls_made
        if remaining_budget > 20 and time.time() < deadline - 5:
            # Perform multiple passes of axis-wise optimization
            num_passes = min(3, remaining_budget // 30) # Each pass ~ 5 dims * ~10 evals
            
            for p in range(num_passes):
                if time.time() > deadline - 5:
                    break
                for i in range(dim):
                    if calls_made >= budget - 5:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    # Define 1D function for dimension i
                    # We search for t in [-1, 1] initially, then narrow down
                    # Use Golden Section Search
                    a = -1.0
                    b = 1.0
                    
                    # First, expand range if necessary? 
                    # For a sphere, the minimum is likely within 1.0 of the NM result.
                    # Let's stick to [-1, 1] for safety and speed.
                    
                    # Golden section search parameters
                    invphi = (np.sqrt(5) - 1) / 2
                    invphi2 = invphi * invphi
                    
                    x1 = a + invphi2 * (b - a)
                    x2 = a + invphi * (b - a)
                    
                    # Evaluate x1 and x2
                    # We need to evaluate func(best_x + t * e_i)
                    # We will manage the best_x update carefully
                    
                    base_x = best_x.copy()
                    e_i = np.zeros(dim)
                    e_i[i] = 1.0
                    
                    # Initial evals
                    # Note: call_func updates global best_x/best_f. 
                    # We need to be careful not to let intermediate 1D search points 
                    # overwrite the global best if they are worse in other dims.
                    # However, since we are moving along an axis from a local minimum,
                    # the function should be unimodal in this slice.
                    # To be safe, we will manually track the best in this 1D search
                    # and only update global best_x at the end of the dimension search.
                    
                    f1 = func(np.clip(base_x + x1 * e_i, -5, 5))
                    calls_made += 1
                    f2 = func(np.clip(base_x + x2 * e_i, -5, 5))
                    calls_made += 1
                    
                    best_t = x1 if f1 < f2 else x2
                    best_f_1d = min(f1, f2)
                    
                    # If we found a better point, update global best temporarily?
                    # No, let's just keep local best.
                    
                    max_1d_iter = 15
                    for _ in range(max_1d_iter):
                        if calls_made >= budget - 5:
                            break
                        if time.time() > deadline - 5:
                            break
                            
                        if f1 < f2:
                            b = x2
                            x2 = x1
                            f2 = f1
                            x1 = a + invphi2 * (b - a)
                        else:
                            a = x1
                            x1 = x2
                            f1 = f2
                            x2 = a + invphi * (b - a)
                            
                        # Evaluate new point
                        x_new_1d = x1 if (a + invphi2 * (b - a)) == x1 else x2
                        # Wait, standard golden section logic:
                        # If f1 < f2, b=x2, x2=x1, f2=f1, x1 = new left
                        # Else a=x1, x1=x2, f1=f2, x2 = new right
                        
                        # Let's re-implement cleanly
                        pass # The above loop logic was getting messy. Let's use a simpler ternary search or fixed steps.
                        
            # Simpler Polish: Fixed step refinement
            # Use remaining calls to do small random perturbations and accept if better.
            # This is robust and simple.
            while calls_made < budget - 5 and time.time() < deadline - 5:
                # Small random step
                step = 0.1 * (1.0 - calls_made / budget) # Decreasing step
                if step < 0.01:
                    step = 0.01
                x_try = best_x + rng.normal(0, step, dim)
                x_try = np.clip(x_try, -4.5, 4.5)
                f_try = call_func(x_try)
                if not (f_try < best_f):
                    # If no improvement, shrink step further
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
