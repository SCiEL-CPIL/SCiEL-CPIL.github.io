"""Multi-start Nelder-Mead ensemble with space-filling initialization and adaptive simplex refinement."""
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
        deadline = start_time + 290.0  # Safety margin for 330s limit
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

        # Generate space-filling starting points using Latin Hypercube Sampling
        # For dim=5, use 5 LHS points (one per dimension stratification)
        n_starts = 5
        lxs = []
        for i in range(n_starts):
            # Each dimension gets a unique stratified position
            pt = np.zeros(dim)
            for d in range(dim):
                # Stratified position: (i + rng.uniform()) / n_starts
                pt[d] = -5 + 10 * ((i + rng.uniform()) / n_starts)
            lxs.append(pt)

        # Also add a few random points for diversity
        for _ in range(3):
            lxs.append(rng.uniform(-5, 5, dim))

        # Evaluate all starting points first
        start_evals = []
        for sp in lxs:
            if not time_ok() or not budget_ok():
                break
            x = np.clip(sp, -5, 5)
            fx = func(x)
            calls += 1
            start_evals.append((x, fx))
            record(x, fx)

        if best_x is None:
            return None, np.inf

        # Sort starting points by function value, best first
        start_evals.sort(key=lambda t: t[1])

        # Launch Nelder-Mead from each of the top starting points
        # Allocate budget proportionally, with more to better starts
        remaining_calls = budget - calls
        n_nm = min(5, len(start_evals))

        # Budget allocation: give more to better starting points
        # Use a decay factor
        weights = []
        for i in range(n_nm):
            weights.append(1.0 / (i + 1))
        total_w = sum(weights)
        allocations = [int(remaining_calls * w / total_w) for w in weights]

        for idx in range(n_nm):
            if not time_ok() or not budget_ok():
                break
            if allocations[idx] < 10:
                continue

            x0, f0 = start_evals[idx]
            x0 = x0.copy()

            # Build initial simplex
            alpha = 0.5
            simplex = [x0.copy()]
            f_simplex = [f0]

            for i in range(dim):
                if not time_ok() or not budget_ok():
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)

            if len(simplex) < 2:
                continue

            # Nelder-Mead parameters
            gamma = 1.0
            rho = 2.0
            sigma_nm = 0.5
            shrink = 0.5

            # Track best in this phase
            phase_best_x = best_x.copy()
            phase_best_f = best_f

            nm_calls = 0
            max_nm_calls = allocations[idx]

            while time_ok() and budget_ok() and nm_calls < max_nm_calls:
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]

                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or not budget_ok():
                    break
                f_r = func(x_r)
                calls += 1
                nm_calls += 1
                record(x_r, f_r)
                if f_r < phase_best_f:
                    phase_best_f = f_r
                    phase_best_x = x_r.copy()

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not time_ok() or not budget_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    nm_calls += 1
                    record(x_e, f_e)
                    if f_e < phase_best_f:
                        phase_best_f = f_e
                        phase_best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)

                    if not time_ok() or not budget_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    nm_calls += 1
                    record(x_c, f_c)
                    if f_c < phase_best_f:
                        phase_best_f = f_c
                        phase_best_x = x_c.copy()

                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or not budget_ok():
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            nm_calls += 1
                            record(simplex[i], f_simplex[i])
                            if f_simplex[i] < phase_best_f:
                                phase_best_f = f_simplex[i]
                                phase_best_x = simplex[i].copy()
                        if not time_ok():
                            break

                # Check for convergence
                f_spread = np.max(f_simplex) - np.min(f_simplex)
                if f_spread < 1e-12 or np.max(np.abs(f_simplex)) < 1e-14:
                    break

                # Also check if simplex has collapsed in space
                max_dist = 0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.max(np.abs(simplex[i] - simplex[j]))
                        max_dist = max(max_dist, d)
                if max_dist < 1e-10:
                    break

        # If we still have budget, do a final round of random perturbations around the best
        if time_ok() and budget_ok():
            for _ in range(10):
                if not time_ok() or not budget_ok():
                    break
                x_new = best_x + rng.normal(0, 0.2, dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                record(x_new, f_new)

        return best_x, best_f
# EVOLVE-BLOCK-END
