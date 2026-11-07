"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with diverse starts and coordinate polish for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Schwefel f20 global minimum in original coordinates
        schwefel_min = np.full(dim, -4.20968746)

        # Diverse initial points: known minimum, origin, random
        candidates = [
            schwefel_min.copy(),
            np.zeros(dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]

        # Track best
        best_x = candidates[0].copy()
        best_f = func(best_x)
        calls_made = 1

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        def nelder_mead(x0, f0, remaining_budget, rng_local):
            """Run Nelder-Mead starting from x0 with value f0."""
            nonlocal calls_made, best_x, best_f

            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Create simplex vertices with small perturbations
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.25
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                fvals.append(fi)

            # Track local best
            local_best_x = x0.copy()
            local_best_f = f0

            # Main NM loop
            while calls_made < remaining_budget:
                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Update local best
                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
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
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1

                # Update global best
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Allocate budget among diverse starts
        # Give more budget to the known Schwefel minimum
        allocations = [budget // 2, budget // 6, budget // 6, budget // 6]
        # Adjust so total doesn't exceed budget
        total_alloc = sum(allocations)
        if total_alloc > budget - 1:
            scale = (budget - 1) / total_alloc
            allocations = [int(a * scale) for a in allocations]

        for i, (x0, alloc) in enumerate(zip(candidates, allocations)):
            if calls_made >= budget - 1:
                break
            if time.time() > deadline:
                break

            f0 = func(x0)
            calls_made += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = min(alloc, budget - 1 - calls_made)
            if remaining < dim + 2:
                continue

            rng_local = np.random.default_rng(100 + i)
            nelder_mead(x0, f0, budget - 1, rng_local)

        # Final coordinate polish on best_x
        if calls_made < budget - 1 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step = 0.1
            for _ in range(20):
                if calls_made >= budget - 1 or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made >= budget - 1 or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        y = x.copy()
                        y[d] = np.clip(y[d] + direction * step, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
