"""Concurrent multi-start Nelder-Mead ensemble with LHS initialization and adaptive simplex refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble: launch independent simplexes from LHS points."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s wall-clock limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Generate space-filling initial points using Latin Hypercube Sampling
        n_starts = 5
        lhs_points = []
        for i in range(dim):
            perm = rng.permutation(n_starts)
            for j in range(n_starts):
                val = -5 + 10 * (perm[j] + 0.5) / n_starts
                val += rng.normal(0, 0.5)
                val = np.clip(val, -5, 5)
                if len(lhs_points) <= j:
                    lhs_points.append([])
                lhs_points[j].append(val)
        
        lhs_points = np.array(lhs_points)

        # Evaluate all starting points
        starts = []
        for i in range(n_starts):
            if not time_ok() or not budget_ok():
                break
            x = np.clip(lhs_points[i], -5, 5)
            fx = func(x)
            calls += 1
            starts.append((x.copy(), fx))
            record(x, fx)

        if best_x is None:
            return None, np.inf

        # Sort starts by function value (best first)
        starts.sort(key=lambda t: t[1])

        # Initialize multiple simplexes
        # Each simplex is a list of (x, f) pairs
        # We'll interleave evaluations from different simplexes
        simplexes = []
        for idx, (x0, f0) in enumerate(starts):
            if not time_ok() or not budget_ok():
                break
            simplex = [(x0.copy(), f0)]
            # Build initial simplex by perturbing each dimension
            alpha = 0.3
            for d in range(dim):
                if not time_ok() or not budget_ok():
                    break
                x_i = x0.copy()
                x_i[d] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append((x_i.copy(), f_i))
                record(x_i, f_i)
            if len(simplex) >= 2:
                simplexes.append(simplex)

        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5

        # Interleave NM iterations across all simplexes
        while time_ok() and budget_ok():
            # For each simplex, do one iteration
            active_simplexes = []
            for simplex in simplexes:
                if not time_ok() or not budget_ok():
                    break
                
                # Sort simplex by function value
                simplex.sort(key=lambda t: t[1])
                x_best, f_best = simplex[0]
                x_worst, f_worst = simplex[-1]
                x_second_worst, f_second_worst = simplex[-2]

                # Centroid of all but worst
                centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)

                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or not budget_ok():
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    record(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = (x_e.copy(), f_e)
                    else:
                        simplex[-1] = (x_r.copy(), f_r)
                elif f_r < f_second_worst:
                    simplex[-1] = (x_r.copy(), f_r)
                else:
                    # Contraction
                    if f_r < f_worst:
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)

                    if not time_ok() or not budget_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = (x_c.copy(), f_c)
                    else:
                        # Shrink
                        new_simplex = [(x_best.copy(), f_best)]
                        for i in range(1, len(simplex)):
                            if not time_ok() or not budget_ok():
                                break
                            x_shrunk = x_best + shrink * (simplex[i][0] - x_best)
                            x_shrunk = np.clip(x_shrunk, -5, 5)
                            f_shrunk = func(x_shrunk)
                            calls += 1
                            record(x_shrunk, f_shrunk)
                            new_simplex.append((x_shrunk.copy(), f_shrunk))
                        simplex = new_simplex

                # Check for convergence
                f_spread = max(p[1] for p in simplex) - min(p[1] for p in simplex)
                max_dist = 0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.max(np.abs(simplex[i][0] - simplex[j][0]))
                        max_dist = max(max_dist, d)
                
                if f_spread < 1e-12 or max_dist < 1e-10:
                    # This simplex has converged; remove it
                    continue
                
                active_simplexes.append(simplex)

            # Update simplexes list with only active ones
            simplexes = active_simplexes

            # If no active simplexes, break
            if not simplexes:
                break

            # If we have budget left and no active simplexes, try a new start
            if not simplexes and time_ok() and budget_ok():
                # Try a random restart
                x_new = rng.uniform(-5, 5, dim)
                f_new = func(x_new)
                calls += 1
                record(x_new, f_new)
                
                simplex = [(x_new.copy(), f_new)]
                alpha = 0.3
                for d in range(dim):
                    if not time_ok() or not budget_ok():
                        break
                    x_i = x_new.copy()
                    x_i[d] += alpha
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls += 1
                    simplex.append((x_i.copy(), f_i))
                    record(x_i, f_i)
                if len(simplex) >= 2:
                    simplexes.append(simplex)

        return best_x, best_f
# EVOLVE-BLOCK-END
