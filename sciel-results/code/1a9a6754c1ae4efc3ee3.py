import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # We will record all evaluated points to compute hypervolume later if needed,
        # but the evaluator records them automatically. We just need to return something.
        # The evaluator ignores the return value.
        
        # Strategy: Multi-Start Local Search
        # 1. Initial random points
        # 2. For each point, perform a local search using finite difference gradient approximation
        #    or simple coordinate descent / Nelder-Mead style steps.
        #    Given the small budget (200) and dim (4), we can afford a decent number of starts.
        
        # Let's allocate budget:
        # 20 starts with 10 evaluations each (1 initial + 9 local steps) = 200 evals.
        # Or 10 starts with 20 evaluations each.
        # Local search in 4D with 2 objectives is tricky. We want to find Pareto-optimal points.
        # A simple approach: Minimize a weighted sum of objectives for different weights.
        # This is a scalarization approach.
        
        # Weights for 2 objectives: (w1, w2) where w1+w2=1.
        # We can sample weights from a distribution.
        
        # However, "Minimizing all 2 objectives" usually implies finding the Pareto front.
        # The scoring is based on Hypervolume of ALL recorded vectors.
        # So we want to populate the Pareto front as well as possible.
        
        # Alternative Structural Change:
        # Use a grid-based or structured sampling combined with local refinement.
        # Or: Use a specific meta-heuristic like Particle Swarm Optimization (PSO) adapted for multi-objective?
        # Or: Use Differential Evolution?
        
        # The directive asks for a MATERIALLY DIFFERENT strategy.
        # Current: NSGA-II (Evolutionary).
        # New: Multi-Start Scalarized Local Search (Gradient-Free).
        
        # For each of K starts:
        # 1. Pick a random point x0.
        # 2. Pick a random weight vector w.
        # 3. Minimize f_w(x) = w1*f1(x) + w2*f2(x) using a local search method (e.g., Nelder-Mead or simple gradient descent with FD).
        # 4. Record all points visited.
        
        # Nelder-Mead is robust and doesn't require gradients.
        # With dim=4, Nelder-Mead uses 5 points in the simplex.
        # Each function evaluation costs 1.
        # A typical NM run might take 20-50 evals to converge.
        # With 200 budget, we can do ~4-10 restarts.
        
        # To ensure diversity on the Pareto front, we should vary the weight vectors w.
        
        def nelder_mead_minimize(func, x0, bounds, max_iter=20):
            """
            Simple Nelder-Mead implementation.
            func: function to minimize (scalar)
            x0: initial point
            bounds: (lo, hi)
            max_iter: max iterations
            Returns list of evaluated points (including x0)
            """
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                # Perturb along axis i
                step = 0.05 * (bounds[1][i] - bounds[0][i]) if bounds[1][i] != bounds[0][i] else 0.05
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            func_vals = [func(x) for x in simplex]
            evaluated_points = [x0.copy()]
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(func_vals)
                simplex = [simplex[i] for i in indices]
                func_vals = [func_vals[i] for i in indices]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                worst_val = func_vals[-1]
                
                # Reflection
                xr = centroid + alpha * (centroid - worst_point)
                xr = np.clip(xr, bounds[0], bounds[1])
                fr = func(xr)
                evaluated_points.append(xr.copy())
                
                if fr < func_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst_point)
                    xe = np.clip(xe, bounds[0], bounds[1])
                    fe = func(xe)
                    evaluated_points.append(xe.copy())
                    
                    if fe < fr:
                        simplex[-1] = xe
                        func_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        func_vals[-1] = fr
                elif fr < func_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    func_vals[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if worst_val < fr, inside otherwise
                    if worst_val < fr:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst_point - centroid)
                    
                    xc = np.clip(xc, bounds[0], bounds[1])
                    fc = func(xc)
                    evaluated_points.append(xc.copy())
                    
                    if fc < min(worst_val, fr):
                        simplex[-1] = xc
                        func_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                            func_vals[i] = func(simplex[i])
                            evaluated_points.append(simplex[i].copy())
                        
                        # Check time again
                        if time.time() > deadline:
                            break
                        
                        if not any(np.allclose(simplex[i], simplex[0]) for i in range(1, dim+1)):
                            continue
                        else:
                            break # Converged or stuck
                
                # Check convergence
                if np.max(func_vals) - np.min(func_vals) < 1e-10:
                    break
            
            return evaluated_points

        # Main Loop
        used = 0
        bounds_tuple = (lo, hi)
        
        # Number of restarts
        # Let's try 5 restarts.
        # Each restart: 1 initial + up to 19 NM iterations (each NM iteration might use 1-3 evals)
        # Actually, let's just run NM with a budget per restart.
        
        n_restarts = 5
        budget_per_restart = budget // n_restarts # 40 evals per restart
        
        for r in range(n_restarts):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Choose weight vector
            # Uniform random weights summing to 1
            w = np.random.random(n_obj)
            w /= np.sum(w)
            
            # Define scalarized objective
            def scalar_obj(x, w=w):
                f = problem(x)
                return np.dot(w, f)
            
            # Initial point
            x0 = lo + (hi - lo) * np.random.random(dim)
            
            # Run Nelder-Mead
            # We need to track evaluations carefully to not exceed budget
            # The NM function above doesn't know about global budget.
            # Let's modify it or wrap it.
            
            # Simpler: Just call problem directly in a loop managed by us?
            # No, let's stick to the NM function but pass a flag or check time/budget inside.
            # Since NM is local, let's just run it.
            
            # To be safe, we'll create a wrapper that counts and stops if budget is hit.
            # But NM needs consistent function values for the simplex.
            # If we stop mid-simplex, it's messy.
            
            # Let's just run NM with a max_iter that roughly fits budget_per_restart.
            # NM uses ~1.5 * dim * max_iter evals on average?
            # Let's set max_iter = budget_per_restart // (dim + 1)
            
            max_iter = max(5, budget_per_restart // (dim + 1))
            
            # We need to evaluate x0 first
            f0 = problem(x0)
            used += 1
            
            # If we run out of budget, break
            if used >= budget:
                break
            
            # The NM function needs to evaluate func(x).
            # We can't easily inject the budget check into the internal loop of NM without refactoring.
            # Let's refactor NM to take a 'evaluator' that can raise an exception or return None?
            # Or just trust the time check inside NM.
            
            # Let's rewrite NM to be budget-aware.
            
            def run_nm_budgeted(x0, w, remaining_budget):
                func_calls = 0
                def func(x):
                    nonlocal func_calls
                    if func_calls >= remaining_budget:
                        raise ValueError("Budget exceeded")
                    func_calls += 1
                    f = problem(x)
                    return np.dot(w, f)
                
                try:
                    pts = nelder_mead_minimize(func, x0, bounds_tuple, max_iter=max_iter)
                    return pts, func_calls
                except ValueError:
                    # Return what we have? The NM function doesn't return partial results easily.
                    # This is a flaw.
                    # Let's just let it run and rely on time check?
                    # Or better: implement a simpler local search that is easier to budget.
                    
                    # Alternative: Coordinate Descent / Hill Climbing
                    # This is easier to control.
                    return None, 0
            
            # Let's switch to a simpler, more controllable local search:
            # Random Direction Hill Climbing
            # 1. Start at x0
            # 2. While budget allows:
            #    a. Pick a random direction d
            #    b. Try step sizes s1, s2, s3
            #    c. If improvement in scalarized objective, move. Else, shrink step.
            
            # This is structurally different from NM and easier to budget.
            
            current_x = x0.copy()
            current_f = f0
            step_size = 0.1 * (hi - lo)
            
            # Local search loop
            while used < budget and time.time() < deadline:
                # Pick random direction
                d = np.random.randn(dim)
                d /= np.linalg.norm(d)
                
                improved = False
                # Try a few step sizes
                for s_factor in [1.0, 0.5, 0.2]:
                    s = s_factor * step_size
                    x_new = current_x + s * d
                    x_new = np.clip(x_new, lo, hi)
                    
                    if np.allclose(x_new, current_x):
                        continue
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    val_new = np.dot(w, f_new)
                    val_curr = np.dot(w, current_f)
                    
                    if val_new < val_curr:
                        current_x = x_new
                        current_f = f_new
                        improved = True
                        break
                    elif used >= budget:
                        break
                
                if not improved:
                    # Shrink step size
                    step_size *= 0.5
                    if step_size < 1e-6 * (hi - lo).max():
                        break
                else:
                    # Maybe grow step size slightly?
                    # step_size *= 1.1
                    pass
            
            # Record the final point? It's already recorded by problem() calls.
            # The evaluator records every call to problem().
            
        return None
