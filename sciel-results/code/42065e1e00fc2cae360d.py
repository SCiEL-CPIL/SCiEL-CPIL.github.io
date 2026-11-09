"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer for the BBOB f24 task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Parameters for NM
        # Initial simplex size
        simplex_size = 0.5
        # Max iterations per restart
        max_iter_per_restart = 50
        # Stagnation threshold
        stagnation_threshold = 1e-6

        # We will use a strategy of multiple short NM runs.
        # Each run consumes up to budget_per_run calls.
        # Let's allocate budget dynamically.
        
        # To ensure we make progress early, we start with a random point and run NM.
        # If it stalls, we restart.
        
        while calls_used < budget:
            # Determine how many calls we can afford for this NM run
            # NM cost per iteration is roughly 1-2 calls.
            # Let's reserve at least 1 call for the initial point evaluation if not done.
            
            # Start a new NM run
            # Pick a starting point. 
            # If we have a previous best, start near it with some noise? 
            # Or start random?
            # For Lunacek, starting random is safer to avoid the trap if the trap is deep.
            # But if we are close to global, we want to exploit.
            # Let's mix: 50% chance start random, 50% chance start near best (if exists).
            
            if best_x is None or rng.random() < 0.5:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb best
                x0 = best_x + rng.normal(0, 1.0, dim)
                x0 = np.clip(x0, -5, 5)

            # Evaluate initial point
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if calls_used >= budget:
                break

            # Build initial simplex
            # Simplex has dim+1 points.
            # We already have x0. We need dim more points.
            simplex = [x0]
            fits = [f0]
            
            for i in range(dim):
                if calls_used >= budget:
                    break
                xi = x0.copy()
                # Move in direction i
                step = simplex_size
                if xi[i] + step > 5:
                    step = -step
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_used += 1
                simplex.append(xi)
                fits.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            if len(simplex) < dim + 1:
                break

            # Nelder-Mead Loop
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter_per_restart):
                if calls_used >= budget:
                    break

                # Sort simplex by fitness
                idx = np.argsort(fits)
                simplex = [simplex[i] for i in idx]
                fits = [fits[i] for i in idx]
                
                # Check convergence
                if abs(fits[-1] - fits[0]) < stagnation_threshold:
                    break

                x_worst = simplex[-1]
                f_worst = fits[-1]
                
                # Centroid of best dim points
                x_c = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fits[0]:
                    # Expansion
                    if calls_used >= budget:
                        break
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fits[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fits[-1] = f_r
                elif f_r < fits[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fits[-1] = f_r
                else:
                    # Contraction
                    if calls_used >= budget:
                        break
                    # Outside contraction if f_r < f_worst, else inside
                    if f_r < f_worst:
                        x_c_out = x_c + rho * (x_r - x_c)
                        x_c_out = np.clip(x_c_out, -5, 5)
                        f_c = func(x_c_out)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c_out.copy()
                        if f_c < f_r:
                            simplex[-1] = x_c_out
                            fits[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                # Re-evaluation of shrink points is expensive.
                                # Standard NM shrinks towards best point.
                                # We need to evaluate them? 
                                # Usually, shrink is applied and the next iteration sorts again.
                                # But we need fitness values.
                                # To save calls, we can just break and restart if shrink is needed frequently.
                                # Or we evaluate them.
                                # Let's evaluate them to maintain simplex integrity.
                                if calls_used >= budget:
                                    break
                                f_shrink = func(simplex[i])
                                calls_used += 1
                                fits[i] = f_shrink
                                if f_shrink < best_f:
                                    best_f = f_shrink
                                    best_x = simplex[i].copy()
                        break # Break after contraction/shrink to re-sort? 
                        # Actually, standard NM replaces worst point.
                        # If we shrink, we replace all points except best.
                        # Let's simplify: if contraction fails, shrink and break loop for this restart.
                    else:
                        x_c_in = x_c + rho * (x_worst - x_c)
                        x_c_in = np.clip(x_c_in, -5, 5)
                        f_c = func(x_c_in)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c_in.copy()
                        if f_c < f_worst:
                            simplex[-1] = x_c_in
                            fits[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_shrink = func(simplex[i])
                                calls_used += 1
                                fits[i] = f_shrink
                                if f_shrink < best_f:
                                    best_f = f_shrink
                                    best_x = simplex[i].copy()
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
