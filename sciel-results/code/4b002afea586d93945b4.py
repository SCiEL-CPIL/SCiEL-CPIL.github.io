import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 330s limit, leave 40s margin
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Phase 1: Initial Sampling (10 points)
        # Spread out initial points to cover the space
        n_init = 10
        for i in range(n_init):
            if not time_ok() or calls_made >= budget:
                break
            # Use a simple stratified or random start
            # Let's use random but ensure they are not too close to each other initially? 
            # Actually, just random is fine for now, or a grid.
            # Let's do a mix: some random, some near origin?
            # Schaffer F7 has a global min. Let's just do random.
            x = rng.uniform(-5, 5, d)
            call_func(x)

        # Phase 2: Nelder-Mead based local search with restarts
        # We will run multiple Nelder-Mead instances from different starting points.
        # Each instance has a budget of evaluations.
        
        # Strategy:
        # 1. Take the current best point.
        # 2. Run Nelder-Mead from it with a limited budget.
        # 3. If it improves, update best.
        # 4. Generate new starting points based on the best (or random if stuck).
        
        # To handle multimodality, we need restarts.
        # Let's allocate budget for restarts.
        # Say, 20 restarts, each with 20 evaluations? 20*20 = 400.
        # Plus initial 10 = 410. Leaves 90 for final polish.
        
        # Actually, Nelder-Mead in 5D takes ~5*(d+1) evaluations per iteration? No, it's d+1 vertices.
        # One iteration uses d+1 evaluations? No, it uses 1 new evaluation per iteration (usually).
        # So 20 evaluations ~ 20 iterations.
        
        # Let's implement a simple Nelder-Mead loop that respects the global budget.
        
        def nelder_mead(start_x, max_iter):
            nonlocal best_x, best_y, calls_made
            
            # Initialize simplex
            # Vertices: start_x, and start_x + step * e_i
            step = 0.5
            simplex = [start_x.copy()]
            for i in range(d):
                v = start_x.copy()
                v[i] += step
                simplex.append(v)
            
            # Evaluate initial simplex
            f_values = []
            for v in simplex:
                if not time_ok() or calls_made >= budget:
                    return
                f_val = call_func(v)
                if f_val is None:
                    return
                f_values.append(f_val)
            
            # Sort simplex
            def sort_simplex():
                nonlocal simplex, f_values
                combined = sorted(zip(f_values, simplex), key=lambda x: x[0])
                f_values = [c[0] for c in combined]
                simplex = [c[1] for c in combined]
            
            sort_simplex()
            
            current_iter = 0
            while current_iter < max_iter and time_ok() and calls_made < budget:
                # Check convergence
                if f_values[0] - f_values[-1] < 1e-8:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = call_func(x_ref)
                if f_ref is None:
                    break
                
                if f_ref < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = call_func(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Simple Reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        rho = 0.5
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        rho = 0.5
                        x_con = centroid + rho * (simplex[-1] - centroid)
                    
                    f_con = call_func(x_con)
                    if f_con is None:
                        break
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if not time_ok() or calls_made >= budget:
                                return
                            f_shrink = call_func(simplex[i])
                            if f_shrink is None:
                                return
                            f_values[i] = f_shrink
                        
                sort_simplex()
                current_iter += 1

        # Main Loop: Restart strategy
        # We will keep a list of "elite" points to restart from.
        # Initially, just the best.
        
        # Let's do 10 restarts of 30 evaluations each?
        # 10 * 30 = 300.
        # Initial 10. Total 310.
        # Remaining 190 for more restarts or longer runs.
        
        # Better: Adaptive budget per restart.
        # Start with small budgets, increase if improving?
        
        # Let's just loop while we have budget.
        # Each restart gets a budget of ~10-20 evaluations initially, increasing?
        # Or fixed 20?
        
        # Let's try: 
        # 1. Run NM from best with 15 evals.
        # 2. If improved, do another 15 from new best.
        # 3. If not improved, generate a new random point (or perturbation of best) and run NM from there with 15 evals.
        
        # This allows global exploration via random restarts and local exploitation via NM.
        
        while time_ok() and calls_made < budget:
            # Determine start point
            # 20% chance to random restart (global exploration)
            # 80% chance to refine best (local exploitation)
            
            if rng.random() < 0.2 and calls_made > 50:
                start_x = rng.uniform(-5, 5, d)
            else:
                # Perturb best slightly? Or just best?
                # If we just use best, NM might get stuck in local min if it's not the global one.
                # But NM from best is good for polishing.
                # Let's use best, but if we've been refining for a while without improvement, force a random restart.
                # Simple heuristic: 
                start_x = best_x.copy()
                # Add small noise to escape exact local minima?
                # start_x += rng.normal(0, 0.1, d)
                
            # Budget for this restart
            # Decrease budget as we go? Or keep it constant?
            # Let's keep it constant at 20 for now.
            # But if we are close to the end, use remaining budget.
            remaining = budget - calls_made
            if remaining < 10:
                break # Not enough for a meaningful restart
            
            restart_budget = min(20, remaining)
            
            # Run Nelder-Mead
            nelder_mead(start_x, restart_budget)
            
            # If we ran out of budget, break
            if calls_made >= budget:
                break

        return best_x, best_y
