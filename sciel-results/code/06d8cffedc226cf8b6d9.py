# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer: Adaptive (1+1)-ES for exploration, Nelder-Mead for exploitation.
    
    Strategy:
    1. Phase 1: (1+1)-ES with restarts to robustly locate the basin of attraction.
       This handles the hidden shift and rotation by exploring broadly.
    2. Phase 2: Nelder-Mead simplex method to precisely converge on the sharp ridge.
       NM adapts to the local anisotropy better than coordinate descent for rotated ridges.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # --- Phase 1: Adaptive (1+1)-ES with Restarts ---
        # Goal: Get into the basin of attraction.
        # Use ~60% of budget for this phase to ensure robust exploration.
        
        window_size = 5
        successes = 0
        steps_in_window = 0
        stall_threshold = 40
        max_sigma = 5.0
        min_sigma = 1e-4
        
        x = np.zeros(n)
        fx = wrapped_func(x)
        sigma = 1.5
        steps_without_improvement = 0
        es_budget_limit = int(budget * 0.6)

        while calls_made[0] < es_budget_limit:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                steps_without_improvement = 0
                sigma = min(sigma * 1.2, max_sigma)
            else:
                steps_without_improvement += 1
                sigma = max(sigma * 0.85, min_sigma)

            steps_in_window += 1
            
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    sigma = max(sigma * 0.9, min_sigma)
                successes = 0
                steps_in_window = 0

            if steps_without_improvement > stall_threshold:
                # Restart: try a random point in a reasonable box
                x = rng.uniform(-3, 3, n)
                fx = wrapped_func(x)
                sigma = 1.5
                steps_without_improvement = 0
                successes = 0
                steps_in_window = 0

        # --- Phase 2: Nelder-Mead Simplex Method ---
        # Goal: Precisely converge on the minimum.
        # NM is effective for sharp ridges as it adapts the simplex shape.
        
        if calls_made[0] < budget and time.time() < deadline:
            # Initialize simplex around the best point found in Phase 1
            x0 = best_x.copy()
            f0 = wrapped_func(x0) # Ensure we have a valid starting point
            
            # Simplex initialization
            # Start with a small perturbation in each dimension
            simplex = [x0.copy()]
            for i in range(n):
                point = x0.copy()
                # Initial step size: 10% of the domain width or a small fixed value
                step = 0.5 
                point[i] += step
                simplex.append(point)
            
            # Evaluate initial simplex
            simplex_f = [f0]
            for i in range(1, n + 1):
                simplex_f.append(wrapped_func(simplex[i]))
            
            # Nelder-Mead Parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma_nm = 0.5 # Shrinkage
            
            # Run NM until budget or time runs out
            while calls_made[0] < budget:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = wrapped_func(xr)
                
                if simplex_f[0] <= fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = wrapped_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    
                    xc = np.clip(xc, -5, 5)
                    fc = wrapped_func(xc)
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrinkage
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = wrapped_func(simplex[i])
                
                # Check for convergence (simple heuristic)
                # If the spread of the simplex is very small, we might be converged
                # But we rely on budget/time limits primarily to avoid overhead
                # A simple check: if max diff in simplex is < 1e-6, break
                if np.max(np.abs(np.array(simplex) - np.mean(simplex, axis=0))) < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
