import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        calls = 0
        x_best = None
        f_best = np.inf

        def nelder_mead(func, x0, max_calls, deadline, current_calls):
            # x0: starting point
            # Returns: x_best_local, f_best_local, calls_made
            x_best_local = x0.copy()
            f_best_local = np.inf
            calls_made = 0
            
            # Create simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            fvals = [np.inf] * (n + 1)
            
            # Initialize simplex
            for i in range(n + 1):
                if i == 0:
                    simplex[i] = x0.copy()
                else:
                    simplex[i] = x0.copy()
                    simplex[i][i-1] += 0.25 * abs(x0[i-1]) + 0.05
                simplex[i] = np.clip(simplex[i], -5, 5)
                fvals[i] = func(simplex[i])
                calls_made += 1
                if fvals[i] < f_best_local:
                    f_best_local = fvals[i]
                    x_best_local = simplex[i].copy()

            # Sort initial simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 100 # Limit iterations
            
            for _ in range(max_iter):
                if calls_made >= max_calls or time.time() > deadline:
                    break
                
                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < f_best_local:
                    f_best_local = f_r
                    x_best_local = x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < f_best_local:
                        f_best_local = f_e
                        x_best_local = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < f_best_local:
                        f_best_local = f_c
                        x_best_local = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < f_best_local:
                                f_best_local = fvals[i]
                                x_best_local = simplex[i].copy()
                        
                        if calls_made >= max_calls or time.time() > deadline:
                            break
                
                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check for convergence
                if np.std(fvals) < 1e-12:
                    break

            return x_best_local, f_best_local, calls_made

        # Initial random sampling
        for _ in range(10):
            if calls >= self.budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, n)
            f = func(x)
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # Run Nelder-Mead from the best point
        if x_best is not None and calls < self.budget:
            x_nm, f_nm, calls_nm = nelder_mead(func, x_best, self.budget - calls, deadline, calls)
            calls += calls_nm
            if f_nm < f_best:
                f_best = f_nm
                x_best = x_nm.copy()

        # If budget remains, try a few more random restarts
        while calls < self.budget and time.time() < deadline:
            # Perturb best point
            x_start = x_best + 0.5 * np.random.randn(n)
            x_start = np.clip(x_start, -5, 5)
            x_nm, f_nm, calls_nm = nelder_mead(func, x_start, self.budget - calls, deadline, calls)
            calls += calls_nm
            if f_nm < f_best:
                f_best = f_nm
                x_best = x_nm.copy()

        return x_best, f_best
