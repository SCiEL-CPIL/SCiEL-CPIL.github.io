import numpy as np
import time

class Optimizer:
    """Adaptive Polytope Subdivision (generalized Nelder-Mead) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 310  # Safety margin before 330s kill
        n = self.dim
        
        # Global best tracking
        best_x_global = np.zeros(n)
        best_f_global = float('inf')
        
        # Initial evaluation at origin
        fx0 = func(np.zeros(n))
        best_f_global = fx0
        best_x_global = np.zeros(n)
        calls = 1

        def time_ok():
            return (time.time() - start_time) < time_limit

        rng = np.random.default_rng(42)

        # Parameters
        max_restarts = 20
        restart_count = 0
        
        # Initial simplex size
        base_alpha = 1.0
        
        while restart_count < max_restarts and calls < self.budget and time_ok():
            # Initialize simplex around best_x_global
            # For the first restart, start at 0. For subsequent, start at best found.
            center = best_x_global.copy()
            
            # Adaptive initial step size: shrink slightly with restarts to refine
            # But if we are stuck, maybe expand? Let's stick to shrinking for exploitation
            # unless we detect complete stagnation, then expand.
            # Actually, let's use a heuristic: if restart_count is 0, alpha=2.0 (large exploration)
            # otherwise alpha=0.5 * (0.8 ** restart_count) to zoom in.
            if restart_count == 0:
                alpha = 2.0
            else:
                alpha = 1.0 * (0.7 ** restart_count)
                if alpha < 0.05:
                    alpha = 0.05

            # Build simplex: n+1 vertices
            # Vertex 0 is the center
            # Vertices 1..n are center +/- alpha along axes
            simplex = np.zeros((n + 1, n))
            simplex[0] = center
            for i in range(n):
                simplex[i + 1] = center
                simplex[i + 1][i] += alpha

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls >= self.budget or not time_ok():
                    break
                fvals[i] = func(simplex[i])
                calls += 1
                if fvals[i] < best_f_global:
                    best_f_global = fvals[i]
                    best_x_global = simplex[i].copy()

            if calls >= self.budget or not time_ok():
                break

            # If we couldn't evaluate all vertices, break
            if calls - (n + 1) > 0 and len([f for f in fvals if np.isfinite(f)]) < n + 1:
                 # This logic is a bit loose, let's just check if we ran out of budget mid-initialization
                 pass

            # Nelder-Mead iterations
            # Standard parameters
            r = 1.0   # Reflection
            ex = 2.0  # Expansion
            c = 0.5   # Contraction
            s = 0.5   # Shrink

            stall_counter = 0
            max_stall = 50  # If no improvement in 50 iterations, restart
            
            prev_best_f = best_f_global
            no_improve_count = 0

            while calls < self.budget and time_ok():
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence / stagnation
                if fvals[0] < prev_best_f - 1e-8:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                prev_best_f = fvals[0]

                if no_improve_count > 20:
                    break

                # Check for small spread
                spread = np.max(np.abs(fvals)) - np.min(np.abs(fvals))
                if spread < 1e-10:
                    break

                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + r * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                if calls >= self.budget or not time_ok():
                    break
                f_r = func(x_r)
                calls += 1
                if f_r < best_f_global:
                    best_f_global = f_r
                    best_x_global = x_r.copy()

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + ex * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    if calls >= self.budget or not time_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f_global:
                        best_f_global = f_e
                        best_x_global = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r < fvals[-1], else inside
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + c * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - c * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    if calls >= self.budget or not time_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f_global:
                        best_f_global = f_c
                        best_x_global = x_c.copy()

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + s * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= self.budget or not time_ok():
                                break
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f_global:
                                best_f_global = fvals[i]
                                best_x_global = simplex[i].copy()
                        
                        if calls >= self.budget or not time_ok():
                            break

                stall_counter += 1
                if stall_counter > 100:
                    break

            restart_count += 1
            # Optional: Random perturbation for next restart center if stuck
            # But we always restart from best_x_global which is good.
            # If best_x_global hasn't changed in a while, we might be in a local min.
            # The alpha shrinkage handles zooming.

        return best_x_global, best_f_global
