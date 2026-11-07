"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Global-Local Optimizer using Nelder-Mead for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            # Clip to box to ensure validity, although func clips too
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Sample diverse points to find a good starting point for local search
        n_global = 30
        for _ in range(n_global):
            if not check_time():
                break
            x = rng.uniform(-5, 5, d)
            evaluate(x)

        # Phase 2: Local Refinement with Nelder-Mead
        # We will run Nelder-Mead from the current best point.
        # If it stagnates, we restart from a new random point or perturb the best.
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Initial step size for simplex
        # If best_x is near boundary, use smaller step
        dist_to_boundary = min(5 - np.max(best_x), np.min(best_x) + 5)
        initial_step = max(0.5, min(1.0, dist_to_boundary))
        
        stagnation_limit = 50
        max_restarts = 5
        restart_count = 0
        
        while check_time() and restart_count < max_restarts:
            # Initialize Simplex
            # x0 is the best point found so far
            x0 = best_x.copy()
            f0 = evaluate(x0)
            if f0 is None: break
            
            simplex = [x0]
            f_simplex = [f0]
            
            # Add d more points
            for i in range(d):
                xi = x0.copy()
                # Use a standard basis vector scaled by initial_step
                # Or random direction? Standard basis is better for axis-aligned problems,
                # but f7 is rotated. Random directions might be better initially.
                # Let's use a mix: first few are basis, then random.
                if i < d:
                    # Standard basis
                    xi[i] += initial_step
                    # If out of bounds, flip or adjust
                    if xi[i] > 5: xi[i] -= 2 * initial_step
                    if xi[i] < -5: xi[i] += 2 * initial_step
                else:
                    # This loop only runs d times, so this else is never hit in this structure
                    pass
                
                fi = evaluate(xi)
                if fi is None: break
                simplex.append(xi)
                f_simplex.append(fi)
            
            if len(simplex) < d + 1:
                break

            # Sort simplex by function value
            def sort_simplex():
                nonlocal simplex, f_simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
            
            sort_simplex()
            
            stagnant = 0
            best_in_run = f_simplex[0]
            
            # Main Nelder-Mead loop
            while check_time():
                sort_simplex()
                best_f_run = f_simplex[0]
                
                if best_f_run < best_in_run:
                    best_in_run = best_f_run
                    stagnant = 0
                else:
                    stagnant += 1
                    
                if stagnant > stagnation_limit:
                    break
                
                # Check convergence
                if max(f_simplex) - min(f_simplex) < 1e-8:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if f_r is None: break
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e is None: break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r < f_simplex[-1] else inside
                    if f_r < f_simplex[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c is None: break
                    
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = evaluate(simplex[i])
                            if f_simplex[i] is None:
                                return best_x, best_f
                        stagnant = stagnation_limit + 1 # Force break
                        break

            # After Nelder-Mead finishes or stagnates
            if not check_time():
                break
            
            # Update global best
            evaluate(best_x) # Ensure best_x is in the pool? No, best_x is tracked in evaluate.
            
            # Prepare for restart
            restart_count += 1
            # Perturb the best point for the next restart
            # Or pick a new random point?
            # If we are stuck, a new random point might be better.
            # But if we found a good basin, perturbing is better.
            # Let's try perturbing first.
            best_x = best_x + 0.5 * rng.standard_normal(d)
            best_x = np.clip(best_x, -5, 5)
            
            # If we've used up most of the budget, stop restarting
            if n_calls > budget * 0.8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
