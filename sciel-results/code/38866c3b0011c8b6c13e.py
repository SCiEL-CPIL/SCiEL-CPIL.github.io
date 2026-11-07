"""Nelder-Mead optimizer with global initialization for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with global initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Global Search to find a point in the attractive sector
        # Evaluate a set of random points and some structured points
        candidates = []
        
        # Origin
        candidates.append(np.zeros(dim))
        
        # Random uniform points
        num_global = 50
        for _ in range(num_global):
            candidates.append(rng.uniform(-5.0, 5.0, size=dim))
            
        # Some axis-aligned probes to help detect sector boundaries if needed
        for i in range(dim):
            for val in [-2.0, 2.0]:
                p = np.zeros(dim)
                p[i] = val
                candidates.append(p)

        best_x = None
        best_f = np.inf
        
        for p in candidates:
            if check_deadline() or calls >= budget:
                break
            p = np.clip(p, -5.0, 5.0)
            fp = func(p)
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Local Optimization using Nelder-Mead
        # Initialize simplex around the best point found
        x0 = best_x
        f0 = best_f
        
        # Initial step size for simplex vertices
        # Use a relatively small step to stay within the sector initially
        step = 0.5
        
        simplex = []
        # Center vertex
        simplex.append((x0.copy(), f0))
        
        # Other vertices
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            simplex.append(xi)
            
        # Evaluate the remaining vertices of the simplex
        for i in range(1, dim + 1):
            if check_deadline() or calls >= budget:
                break
            xi = simplex[i][0]
            xi = np.clip(xi, -5.0, 5.0)
            fi = func(xi)
            calls += 1
            simplex[i] = (xi, fi)

        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # Track best
        current_best_x = best_x
        current_best_f = best_f

        # Main Nelder-Mead loop
        max_iter = budget - calls
        for _ in range(max_iter):
            if check_deadline() or calls >= budget:
                break

            # Sort simplex by function value
            simplex.sort(key=lambda x: x[1])
            
            # Check if we found a better point
            if simplex[0][1] < current_best_f:
                current_best_f = simplex[0][1]
                current_best_x = simplex[0][0].copy()

            # Identify worst, second worst, and best points
            x_worst = simplex[-1][0]
            f_worst = simplex[-1][1]
            x_second_worst = simplex[-2][0]
            x_best = simplex[0][0]
            f_best = simplex[0][1]
            
            # Calculate centroid of all points except the worst
            x_centroid = np.sum([simplex[i][0] for i in range(dim)], axis=0) / dim
            
            # 1. Reflection
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < f_best:
                # 2. Expansion
                x_exp = x_centroid + gamma * (x_ref - x_centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    x_new = x_exp
                    f_new = f_exp
                else:
                    x_new = x_ref
                    f_new = f_ref
            elif f_ref < simplex[-2][1]:
                # Accept reflection
                x_new = x_ref
                f_new = f_ref
            else:
                # 3. Contraction
                if f_ref < f_worst:
                    # Outside contraction
                    x_con = x_centroid + rho * (x_ref - x_centroid)
                else:
                    # Inside contraction
                    x_con = x_centroid - rho * (x_ref - x_centroid)
                
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = func(x_con)
                calls += 1
                
                if f_con < min(f_ref, f_worst):
                    x_new = x_con
                    f_new = f_con
                else:
                    # 4. Shrink
                    # Shrink all points (except the best) towards the best
                    # We replace the worst point with a shrunk version
                    # Standard NM shrinks all points except the best towards the best.
                    # However, for efficiency, we can just shrink the worst point towards the best.
                    # Let's implement the standard shrink: replace all vertices except the best with midpoint to best.
                    # This requires dim evaluations.
                    # To save budget, we might just do a smaller step in the direction of the best.
                    # Let's do a simple shrink of the worst point.
                    x_shrink = x_best + sigma * (x_worst - x_best)
                    x_shrink = np.clip(x_shrink, -5.0, 5.0)
                    f_shrink = func(x_shrink)
                    calls += 1
                    
                    if f_shrink < f_worst:
                        x_new = x_shrink
                        f_new = f_shrink
                    else:
                        # If shrink fails, we are stuck. 
                        # We can try a random restart or reduce step size.
                        # For now, just keep the worst point but maybe perturb it?
                        # Or break if no improvement for many steps?
                        # Let's just accept the worst point to continue, but this is inefficient.
                        # Better: Break if we are stuck for a long time? 
                        # Let's just continue, the sorting will handle it.
                        x_new = x_worst
                        f_new = f_worst

            # Replace the worst point in the simplex
            simplex[-1] = (x_new, f_new)

        return current_best_x, current_best_f
# EVOLVE-BLOCK-END
