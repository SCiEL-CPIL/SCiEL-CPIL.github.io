import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 330s limit, leave 35s margin
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0
        all_points = []
        all_values = []

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            all_points.append(x.copy())
            all_values.append(y)
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Phase 1: Initial Sampling
        # Use a mix of random and structured sampling to cover the space.
        n_init = 20
        # Some random
        for _ in range(n_init // 2):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)
        
        # Some structured (grid-like but perturbed)
        # In 5D, a full grid is too large. Use a sparse set.
        # Let's just do more random but with a different seed or distribution.
        for _ in range(n_init // 2):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)

        # Phase 2: Main Loop - Adaptive CMA-ES like strategy
        # Since we don't have a built-in CMA-ES in stdlib/np/scipy that we can easily
        # integrate with strict budget control in a simple loop, we will implement
        # a simplified version or use a robust local search method like
        # Powell's method or a custom gradient-free optimizer.
        
        # However, for robustness and simplicity within the constraints,
        # a well-tuned Nelder-Mead with smart restarts is often competitive.
        # But the directive suggests a structural change.
        # Let's try a simple Covariance Matrix Adaptation (CMA) like approach
        # or just a more aggressive multi-start local search.
        
        # Actually, let's implement a simple version of CMA-ES or use
        # scipy.optimize.minimize with different methods if available?
        # No, the directive says "Python standard library, NumPy and SciPy are available".
        # We can use scipy.optimize.minimize.
        
        # Let's try a hybrid:
        # 1. Initial sampling.
        # 2. Run multiple local optimizations (e.g., Powell or Nelder-Mead) from the best points.
        # 3. Use the results to guide further sampling.
        
        # Since we need to be efficient and robust, let's use a population-based
        # approach with local refinement.
        
        # Define a local optimizer function
        def local_optimize(start_x, max_iter=50):
            nonlocal best_x, best_y, calls_made
            
            # Use Nelder-Mead via scipy or manual?
            # Manual NM is safer for budget control.
            # Let's use a manual NM with adaptive step size.
            
            # Initialize simplex
            step = 0.5
            simplex = [start_x.copy()]
            for i in range(d):
                v = start_x.copy()
                v[i] += step
                simplex.append(v)
            
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
                if f_values[0] - f_values[-1] < 1e-10:
                    break
                
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

        # Main Loop
        # Keep a list of elite points.
        elites = [best_x.copy()]
        
        while time_ok() and calls_made < budget:
            # Select a point to refine
            # With some probability, pick a random point to explore
            if rng.random() < 0.1 and len(all_points) > 10:
                # Pick a random point from history
                idx = rng.integers(0, len(all_points))
                start_x = all_points[idx].copy()
            elif len(elites) > 1 and rng.random() < 0.5:
                # Pick a random elite
                idx = rng.integers(0, len(elites))
                start_x = elites[idx].copy()
            else:
                # Pick the best
                start_x = best_x.copy()
            
            # Add some noise to the start point to avoid re-optimizing the exact same spot
            noise = rng.normal(0, 0.1, d)
            start_x = np.clip(start_x + noise, -5, 5)
            
            # Run local optimization
            remaining = budget - calls_made
            if remaining < 10:
                break
            
            # Adaptive budget for local optimization
            # If we are close to the optimum, use more budget.
            # Simple heuristic: fixed budget of 20-30.
            local_budget = min(25, remaining)
            
            local_optimize(start_x, local_budget)
            
            # Update elites
            if best_x is not None:
                # Add best_x to elites if it's new
                is_new = True
                for e in elites:
                    if np.linalg.norm(best_x - e) < 1e-5:
                        is_new = False
                        break
                if is_new:
                    elites.append(best_x.copy())
                    # Keep only top 5 elites
                    if len(elites) > 5:
                        # Sort elites by their current best value (we don't have individual values, so just keep recent ones)
                        # Better: keep the best 5 based on the value at the time they were added?
                        # For simplicity, just keep the last 5.
                        elites = elites[-5:]

        return best_x, best_y
