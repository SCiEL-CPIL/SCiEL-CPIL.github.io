"""Optimized program for BBOB f21 (Gallagher 101 peaks) with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        n_calls = 0
        best_f = float('inf')
        best_x = None
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return 1e18
            n_calls += 1
            return func(x)

        def nelder_mead(x0, max_calls, rng):
            nonlocal n_calls, best_f, best_x
            
            if n_calls >= budget:
                return
            
            # Initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0
                simplex[i, i-1] += 0.5 # Initial step size 0.5
            
            # Evaluate initial simplex
            f_simplex = np.zeros(dim + 1)
            for i in range(dim + 1):
                if n_calls >= budget:
                    return
                f_simplex[i] = wrapped_func(simplex[i])
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Main loop
            while n_calls < budget:
                # Sort simplex by function value
                idx = np.argsort(f_simplex)
                simplex = simplex[idx]
                f_simplex = f_simplex[idx]
                
                # Centroid of the best dim points
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if n_calls >= budget:
                    break

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = wrapped_func(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if n_calls < budget:
                                    f_simplex[i] = wrapped_func(simplex[i])
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        x_c = np.clip(x_c, -5, 5)
                        f_c = wrapped_func(x_c)
                        if f_c < f_simplex[-1]:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if n_calls < budget:
                                    f_simplex[i] = wrapped_func(simplex[i])
                
                # Check for convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                    break

        rng = np.random.default_rng(42)
        
        # Multi-start strategy
        # f21 has 101 peaks, so a good spread is needed.
        # We allocate a budget for each start.
        # Let's try 10 starts with 49 calls each (1 initial + 48 NM steps approx).
        # Actually NM steps cost 1 or 2 calls per iteration.
        # 500 calls / 10 starts = 50 calls per start.
        
        n_starts = 10
        calls_per_start = budget // n_starts
        
        for i in range(n_starts):
            if n_calls >= budget or time.time() > deadline:
                break
            
            # Random start point
            x0 = rng.uniform(-4.5, 4.5, dim)
            
            nelder_mead(x0, calls_per_start, rng)
            
            # If we have budget left and haven't used all starts, continue
            # If budget is tight, we might just do fewer starts with more calls each
            # But for f21, diversity is key.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
