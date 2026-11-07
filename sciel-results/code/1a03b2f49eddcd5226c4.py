"""Rewritten optimizer for BBOB-style f14 (Different powers) task."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    Uses a derivative-free local search strategy (Nelder-Mead) with multiple
    random initializations to handle non-convexity and rotations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        # Helper to wrap func and count calls
        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f 
            calls_used += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # 1. Initial random scan to find diverse starting points
        # Use 20% of budget for random search
        num_random = max(10, budget // 5)
        candidates = []
        for _ in range(num_random):
            if calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = wrapped_func(x)
            candidates.append((f, x))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # 2. Multi-start Nelder-Mead
        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f
            
        # Run Nelder-Mead from the top few candidates
        num_runs = min(5, len(candidates))
        # Each run gets a portion of the remaining budget
        # We don't strictly enforce budget per run, but we check global budget
        
        for i in range(num_runs):
            if calls_used >= budget:
                break
            
            start_x = candidates[i][1].copy()
            start_f = candidates[i][0]
            
            # Initialize simplex
            # Center: start_x
            # Vertices: start_x +/- delta * e_i
            delta = 0.5
            simplex = [start_x.copy()]
            for d in range(dim):
                x_plus = start_x.copy()
                x_plus[d] += delta
                simplex.append(x_plus)
                
            # Evaluate initial simplex
            # start_x is already evaluated, so we only need to evaluate the other dim points
            if calls_used + dim > budget:
                break
                
            fvals = [start_f]
            for d in range(dim):
                fvals.append(wrapped_func(simplex[d+1]))
                
            # Nelder-Mead Loop
            max_iter = 50
            for _ in range(max_iter):
                if calls_used >= budget:
                    break
                    
                # Sort simplex by fitness
                combined = list(zip(fvals, simplex))
                combined.sort(key=lambda t: t[0])
                fvals = [c[0] for c in combined]
                simplex = [c[1] for c in combined]
                
                # Check convergence
                if fvals[-1] - fvals[0] < 1e-8:
                    break
                    
                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = wrapped_func(x_r)
                
                if f_r < fvals[-2]:
                    # Expansion
                    if f_r < fvals[0]:
                        gamma = 2.0
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        f_e = wrapped_func(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fvals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        rho = 0.5
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = wrapped_func(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                fvals[j] = wrapped_func(simplex[j])
                    else:
                        # Inside contraction
                        rho = 0.5
                        x_c = centroid - rho * (centroid - simplex[-1])
                        f_c = wrapped_func(x_c)
                        if f_c < fvals[-1]:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for j in range(1, dim + 1):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                fvals[j] = wrapped_func(simplex[j])
                            
        # 3. Final random exploration if budget remains
        while calls_used < budget:
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
