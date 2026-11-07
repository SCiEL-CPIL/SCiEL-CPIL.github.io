"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive step size and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        tol_abs = 1e-10
        tol_rel = 1e-8

        def nelder_mead(x0, f0, init_sigma):
            nonlocal calls, best_x, best_f
            
            if calls >= budget:
                return x0, f0

            # Initialize simplex with adaptive initial size
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    return x0, f0
                xi = x0.copy()
                xi[i] += init_sigma
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                fvals.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            max_iter = 3000
            no_improve_count = 0
            prev_best_local = best_f

            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Check convergence - use both absolute and relative
                f_range = max(fvals) - min(fvals)
                f_scale = max(abs(min(fvals)), 1e-8)
                if f_range < tol_abs or f_range / f_scale < tol_rel:
                    break

                # Track local improvement
                if best_f < prev_best_local - 1e-10:
                    no_improve_count = 0
                    prev_best_local = best_f
                else:
                    no_improve_count += 1

                # If stuck for too long, break to allow restart
                if no_improve_count > 200:
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
                            simplex[i] = np.clip(simplex[i], -5, 5)
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

        def polish(x0, f0, max_calls=50):
            """Final high-precision polish with very small steps."""
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return x0, f0
            
            # Use coordinate descent with very small steps for final polish
            x = x0.copy()
            f = f0
            step_size = 0.01
            for coord in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                for _ in range(10):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Try positive step
                    x_plus = x.copy()
                    x_plus[coord] += step_size
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = func(x_plus)
                    calls += 1
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    else:
                        # Try negative step
                        x_minus = x.copy()
                        x_minus[coord] -= step_size
                        x_minus = np.clip(x_minus, -5, 5)
                        f_minus = func(x_minus)
                        calls += 1
                        if f_minus < f:
                            x = x_minus
                            f = f_minus
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                    step_size *= 0.5
            return x, f

        # Multi-start strategy with adaptive restarts and coarse-to-fine schedule
        
        # First start at 0 - good heuristic for rotated Rosenbrock
        x_start = np.zeros(n)
        f_start = func(x_start)
        calls += 1
        best_x = x_start.copy()
        best_f = f_start

        if calls >= budget:
            return best_x, best_f

        # Coarse-to-fine: start with larger simplex for exploration
        x_curr, f_curr = nelder_mead(x_start, f_start, init_sigma=1.0)
        
        # Adaptive restart with decreasing simplex size
        num_restarts_done = 1
        max_restarts = 8
        current_sigma = 1.0
        
        for restart_idx in range(max_restarts - 1):
            if calls >= budget or time.time() > deadline:
                break
            
            # Decrease simplex size for exploitation
            current_sigma *= 0.5
            if current_sigma < 0.01:
                current_sigma = 0.01
            
            # Generate starting point near best found with some randomness
            if best_f < 1.0:
                # Exploitation: start near best
                x_new = best_x + rng.normal(0, current_sigma * 0.5, n)
            else:
                # Exploration: random point
                x_new = rng.uniform(-5, 5, n)
            x_new = np.clip(x_new, -5, 5)
            
            f_new = func(x_new)
            calls += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            if calls >= budget:
                break

            # Run Nelder-Mead from this point
            x_curr, f_curr = nelder_mead(x_new, f_new, current_sigma)
            num_restarts_done += 1
            
            # If we've converged well, do final polish and stop
            if best_f < 1e-4:
                if calls < budget - 60 and time.time() < deadline:
                    x_polished, f_polished = polish(best_x, best_f)
                break

        # Final polish if we have budget remaining
        if calls < budget - 20 and time.time() < deadline and best_f < 10.0:
            x_polished, f_polished = polish(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
