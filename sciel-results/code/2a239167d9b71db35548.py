"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        tol = 1e-8     # tolerance for convergence
        
        # Initial simplex size
        init_sigma = 1.0

        def nelder_mead(x0, f0, budget_left):
            nonlocal calls, best_x, best_f
            
            if budget_left <= 0:
                return x0, f0

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                xi = x0.copy()
                xi[i] += init_sigma
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                fvals.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                
                if budget_left <= 0:
                    return x0, f0

            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            max_iter = 1000  # Safety cap for iterations per restart
            
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Check convergence
                if max(fvals) - min(fvals) < tol:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if calls >= budget:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    
                    if calls >= budget:
                        break

                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if calls >= budget:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break

                # Re-sort simplex
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]

            return simplex[0], fvals[0]

        # Multi-start strategy
        # We will run Nelder-Mead from several starting points.
        # Since Rosenbrock has a narrow valley, starting near the global minimum 
        # (which is hidden) is hard. However, starting near 0 is a good heuristic 
        # for rotated Rosenbrock if the rotation is not too extreme, or we can 
        # use a coarse grid or random starts.
        
        # Given budget 500, we can afford a few restarts.
        # Let's try 5 restarts.
        num_restarts = 5
        points_per_restart = budget // num_restarts
        
        # First start at 0
        x_start = np.zeros(n)
        f_start = func(x_start)
        calls += 1
        best_x = x_start.copy()
        best_f = f_start

        if calls >= budget:
            return best_x, best_f

        # Run Nelder-Mead from 0
        x_curr, f_curr = nelder_mead(x_start, f_start, budget - calls)
        
        # Subsequent restarts from random points
        for _ in range(num_restarts - 1):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate a new random starting point
            # Use a wider distribution to explore the space
            x_new = rng.uniform(-5, 5, n)
            f_new = func(x_new)
            calls += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            if calls >= budget:
                break

            # Run Nelder-Mead from this point
            x_curr, f_curr = nelder_mead(x_new, f_new, budget - calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
