"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Multi-start Nelder-Mead followed by (1+1)-ES polish.
    Designed for BBOB f13 (Sharp Ridge).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        deadline = start_time + 290.0
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Wrapper to handle budget and time limits
        def safe_call(x):
            nonlocal calls_made
            if time.time() > deadline:
                return None
            if calls_made >= self.budget:
                return None
            calls_made += 1
            return func(x)

        # Strategy:
        # 1. Multi-start Nelder-Mead to find a good basin/ridge.
        # 2. (1+1)-ES to polish the best point.
        
        # Determine allocation
        # If budget is small, reduce starts.
        if self.budget < 150:
            n_starts = 1
            nm_budget = int(self.budget * 0.6)
        elif self.budget < 300:
            n_starts = 2
            nm_budget = int(self.budget * 0.35)
        else:
            n_starts = 3
            nm_budget = 100 # 3 * 100 = 300, leaving 200 for ES

        for i in range(n_starts):
            # Random start within [-5, 5]
            # Using a fixed seed per instance index is not possible as we don't know index, 
            # but np.random default is fine.
            x0 = np.random.uniform(-5, 5, self.dim)
            
            x_best, f_best = self._nelder_mead(
                safe_call, x0, nm_budget, deadline
            )
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
            # Check if we should stop early
            if calls_made >= self.budget or time.time() > deadline:
                break

        # Phase 2: Polish with (1+1)-ES
        if best_x is not None:
            remaining_budget = self.budget - calls_made
            if remaining_budget > 0:
                x_best, f_best = self._es_polish(
                    safe_call, best_x, best_f, remaining_budget, deadline
                )
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best

        # If no valid point was found (should not happen), return a default
        if best_x is None:
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             # Note: this call might exceed budget if we were at the limit, 
             # but safe_call would have returned None. 
             # We assume best_x is not None because we made at least one call in NM.

        return best_x, best_f

    def _nelder_mead(self, safe_call, x0, max_evals, deadline):
        """
        Nelder-Mead implementation.
        Returns (best_x, best_f).
        """
        dim = self.dim
        # Initial simplex: x0 and x0 + delta * e_i
        delta = 2.0 # Large initial step to cover ridge width
        
        # Construct simplex
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            simplex.append(v)
        
        # Evaluate initial simplex
        f_values = []
        for v in simplex:
            f = safe_call(v)
            if f is None:
                # Time or budget exceeded
                break
            f_values.append(f)
            
        if len(f_values) < dim + 1:
             # Couldn't even complete initial simplex
             # Return best found so far
             if f_values:
                 idx = np.argmin(f_values)
                 return simplex[idx], f_values[idx]
             return x0, float('inf')

        best_x = simplex[np.argmin(f_values)]
        best_f = min(f_values)

        # Main loop
        # We track number of evaluations used
        evals_used = len(f_values)
        
        # NM parameters
        alpha = 1.0 # Reflection
        gamma = 2.0 # Expansion
        rho = 0.5   # Contraction
        sigma = 0.5 # Shrink
        
        # Iterate
        while evals_used < max_evals:
            # Sort simplex by f-value
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Check convergence: spread of function values
            f_min = f_values[0]
            f_max = f_values[-1]
            if abs(f_max - f_min) < 1e-8:
                break
                
            # Check simplex size
            centroid = np.mean(simplex, axis=0)
            # Simple size metric: max distance from centroid
            if np.max(np.linalg.norm(np.array(simplex) - centroid, axis=1)) < 1e-6:
                break

            # Reflection
            x_worst = simplex[-1]
            x_centroid = np.mean(simplex[:-1], axis=0)
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            
            f_r = safe_call(x_r)
            if f_r is None:
                break
            evals_used += 1
            
            if f_r < f_values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_worst)
                f_e = safe_call(x_e)
                if f_e is None:
                    break
                evals_used += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outer contraction
                    x_c = x_centroid + rho * (x_r - x_worst)
                else:
                    # Inner contraction
                    x_c = x_centroid - rho * (x_r - x_worst)
                
                f_c = safe_call(x_c)
                if f_c is None:
                    break
                evals_used += 1
                
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_shrunk = safe_call(simplex[i])
                        if f_shrunk is None:
                            break
                        evals_used += 1
                        f_values[i] = f_shrunk
                        if f_shrunk is None:
                             break

            # Update best
            current_min_idx = np.argmin(f_values)
            if f_values[current_min_idx] < best_f:
                best_f = f_values[current_min_idx]
                best_x = simplex[current_min_idx].copy()

        return best_x, best_f

    def _es_polish(self, safe_call, x, f, max_evals, deadline):
        """
        (1+1)-ES polish.
        """
        dim = self.dim
        sigma = 0.5
        success = 0
        total = 0
        
        for _ in range(max_evals):
            if time.time() > deadline:
                break
            
            y = x + sigma * np.random.standard_normal(dim)
            fy = safe_call(y)
            if fy is None:
                break
            
            if fy < f:
                x = y
                f = fy
                success += 1
            total += 1
            
            # Adapt sigma
            if total > 20:
                rate = success / total
                if rate > 0.3:
                    sigma *= 1.2
                elif rate < 0.1:
                    sigma *= 0.8
                # Reset stats
                success = 0
                total = 0

        return x, f
# EVOLVE-BLOCK-END
