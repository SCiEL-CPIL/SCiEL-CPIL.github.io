import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard: evaluator kills at 330s, leave 40s safety margin
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x_start, initial_step=0.5):
            """Nelder-Mead for local refinement."""
            nonlocal evals_used, best_f, best_x, deadline
            
            if time.time() > deadline or evals_used >= budget:
                return
            
            # Initialize simplex
            step = initial_step
            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            fs = []
            for i in range(dim + 1):
                if evals_used >= budget or time.time() > deadline:
                    return
                f_val = try_func(simplex[i])
                fs.append(f_val)
                
                # Update global best immediately
                if f_val < best_f:
                    best_f = f_val
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha_nm = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            # Run until budget/time exhausted or converged
            while evals_used < budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check convergence
                if fs[-1] - fs[0] < 1e-10:
                    break
                
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                f_ref = try_func(x_ref)
                
                if np.isinf(f_ref):
                    break
                    
                if f_ref < fs[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = try_func(x_exp)
                    if np.isinf(f_exp):
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                    elif f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = try_func(x_con)
                    if np.isinf(f_con):
                        break
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            fs[i] = try_func(simplex[i])
                            if np.isinf(fs[i]):
                                break
                        if any(np.isinf(f) for f in fs):
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = simplex[i].copy()

        # Phase 1: Exploration to find diverse promising regions
        # We will use 20 evaluations for exploration
        num_explore = 20
        candidates = []
        
        # Always try center
        f_center = try_func(np.zeros(dim))
        if not np.isinf(f_center):
            candidates.append((f_center, np.zeros(dim)))
            
        for i in range(num_explore - 1):
            if evals_used >= budget or time.time() > deadline:
                break
            # Sample from the box, slightly smaller than [-5,5] to avoid boundary issues if possible
            x = rng.uniform(-4.5, 4.5, size=dim)
            f_val = try_func(x)
            if not np.isinf(f_val):
                candidates.append((f_val, x))
        
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        if not candidates:
            # Fallback if all failed
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Select top 10 distinct points (or fewer if not enough)
        # To ensure diversity, we might want to pick top K, but if they are all clustered, 
        # we might miss other basins. 
        # For f7 (step ellipsoidal), the landscape is smooth except for the step.
        # Multi-start from the best found points is a safe bet.
        num_starts = 10
        starts = [c[1] for c in candidates[:num_starts]]
        
        # If we have fewer than num_starts valid candidates, pad with random ones from exploration
        while len(starts) < num_starts and evals_used < budget:
             x = rng.uniform(-4.5, 4.5, size=dim)
             f_val = try_func(x)
             if not np.isinf(f_val):
                 starts.append(x)
        
        # Phase 2: Multi-start Nelder-Mead
        # Distribute remaining budget roughly equally among starts, 
        # but allow early termination if a start converges or budget runs out.
        
        # We will iterate through starts. 
        # A simple strategy: run NM on start 1, then start 2, etc.
        # However, if start 1 is very good, it might use all budget.
        # To be fair, we can cap the number of evaluations per NM run?
        # Or just let them run until budget is gone. 
        # Given the "early progress counts" metric, finding *a* good solution fast is key.
        # So running the best start first is good.
        
        # To prevent one bad start from wasting too much time, 
        # we can set a max iterations or eval limit per start?
        # With 500 budget and 10 starts, 50 evals each is a reasonable cap for exploration, 
        # but NM converges fast. 
        # Let's just run them sequentially. The first one (best candidate) will likely 
        # converge quickly or find the global min.
        
        for start_point in starts:
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Use a smaller initial step for refinement
            # The exploration already found the basin, so 0.5 or 1.0 is fine.
            # Since we are in a box of size 10, and minimizer is in [-4,4], 
            # and we are starting from a point in that basin, 0.5 is safe.
            nelder_mead(start_point, initial_step=0.5)
            
            # If we found a very good solution, we might stop early?
            # Hard to know "very good" without knowing f_opt.
            # But if the simplex has converged (function values identical), NM will return.
        
        return best_x, best_f
