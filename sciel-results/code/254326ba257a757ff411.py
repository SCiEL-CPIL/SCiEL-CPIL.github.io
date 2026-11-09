"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Multi-Start Nelder-Mead with strict budget control."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Helper to evaluate and track
        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Global Exploration via Random Sampling
        # Use 20% of budget for random sampling to find diverse basins
        n_random = self.budget // 5  # 100 samples
        if n_random < 10:
            n_random = 10
            
        random_points = rng.uniform(-5, 5, (n_random, self.dim))
        random_vals = np.array([eval_func(x) for x in random_points])
        
        # Sort by value
        sorted_idx = np.argsort(random_vals)
        
        # Phase 2: Local Refinement via Nelder-Mead on top candidates
        # Use remaining 80% of budget
        remaining_budget = self.budget - call_count
        
        # We will run NM on the top 5 unique good points
        top_k = 5
        # Estimate calls per NM run. NM uses n+1 points initially, then 1-2 per iteration.
        # With dim=5, initial simplex is 6 points.
        # Let's allocate a fixed budget per local search, say 50 calls each.
        # 5 * 50 = 250 calls. This fits within remaining 400.
        
        # To strictly control calls in scipy's minimize, we don't have a perfect hook.
        # However, we can implement a simple Nelder-Mead loop ourselves for strict control.
        # Or, we can trust that if we return inf, it might stop? No, it might loop.
        
        # Let's implement a simple, robust Nelder-Mead variant with strict call limits.
        # This avoids scipy's black-box nature regarding budget.
        
        def nelder_mead_local(x0, max_calls):
            nonlocal call_count, best_x, best_f
            
            n = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [eval_func(x0)]
            
            # Expand simplex
            for i in range(n):
                xi = x0.copy()
                step = 0.5 if xi[i] == 0 else 0.5 * abs(xi[i])
                # Ensure step is reasonable
                step = max(0.1, min(1.0, step))
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                f_simplex.append(eval_func(xi))
                
                if call_count >= self.budget or time.time() > deadline:
                    return
            
            # Iterative improvement
            # We stop when max_calls for this local search is reached
            calls_in_this_local = max_calls
            # We already used n+1 calls for initialization.
            # Let's just run for a fixed number of iterations or until call budget for this block is hit.
            # A simpler heuristic: run until we've used ~max_calls total for this restart.
            # Since we can't easily count inside scipy, let's do a manual NM loop.
            
            # Sort simplex
            for _ in range(max_calls):
                if call_count >= self.budget or time.time() > deadline:
                    break
                    
                # Sort by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check convergence (simple: if best is good enough or no improvement)
                if f_simplex[0] - f_simplex[-1] < 1e-8:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                
                if f_simplex[0] <= fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                elif fr < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                        
                    # Contraction (outside)
                    # If fe >= fr, we already accepted reflection. 
                    # If fe >= f_simplex[-2], we might need contraction. 
                    # For simplicity, let's just do standard contraction if reflection is bad.
                else:
                    # Contraction
                    rho = 0.5
                    if fr < f_worst:
                        # Inside contraction
                        xc = centroid + rho * (worst - centroid)
                    else:
                        # Outside contraction
                        xc = centroid - rho * (worst - centroid)
                        
                    xc = np.clip(xc, -5, 5)
                    fc = eval_func(xc)
                    
                    if fc < min(fr, f_worst):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = eval_func(simplex[i])
                            if call_count >= self.budget or time.time() > deadline:
                                return

        # Run local search on top 5 points from random sampling
        for i in range(top_k):
            if call_count >= self.budget or time.time() > deadline:
                break
            
            # Allocate remaining budget evenly to remaining local searches
            remaining_searches = top_k - i
            calls_per_search = (self.budget - call_count) // remaining_searches
            if calls_per_search < 10:
                calls_per_search = 10
            
            x0 = random_points[sorted_idx[i]]
            nelder_mead_local(x0, calls_per_search)

        # Phase 3: If budget remains, do a final polish on the global best
        if call_count < self.budget and time.time() < deadline:
            # Use remaining budget for one more NM run on the best found so far
            remaining = self.budget - call_count
            nelder_mead_local(best_x, remaining)

        return best_x, best_f

# EVOLVE-BLOCK-END
