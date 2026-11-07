"""Multi-start Nelder-Mead simplex for f24 (Lunacek bi-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    """
    Multi-start Nelder-Mead derivative-free optimizer for Lunacek bi-Rastrigin.
    Uses diverse initial points informed by the function's known basin structure,
    with adaptive simplex parameters and final coordinate polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        # Diverse starting points informed by f24 structure:
        # f24 has two main basins; we sample around known interesting regions
        # plus random points across the domain
        base_starts = [
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        for _ in range(8):
            base_starts.append(rng.uniform(-4.5, 4.5, dim))

        n_starts = len(base_starts)
        total_evals_available = budget - n_starts

        weights = [0.25, 0.25, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]
        weights = np.array(weights[:n_starts])
        weights = weights / weights.sum()
        budgets_per_start = (total_evals_available * weights).astype(int)
        budgets_per_start = np.maximum(budgets_per_start, 10)
        while budgets_per_start.sum() > total_evals_available and budgets_per_start.min() > 10:
            idx = np.argmax(budgets_per_start)
            budgets_per_start[idx] -= 1

        def nelder_mead(start_x, max_evals):
            nonlocal best_x, best_f, calls_made

            if calls_made[0] >= budget or time.time() > deadline:
                return

            dim_local = dim
            n_simplex = dim_local + 1

            simplex = np.empty((n_simplex, dim_local))
            simplex_f = np.empty(n_simplex)

            simplex[0] = start_x.copy()
            fx = counted_func(simplex[0])
            simplex_f[0] = fx
            if fx < best_f:
                best_f = fx
                best_x = simplex[0].copy()

            alpha = 0.1
            for i in range(1, n_simplex):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                simplex[i] = start_x.copy()
                simplex[i][i - 1] += alpha
                simplex[i] = np.clip(simplex[i], -5, 5)
                fx = counted_func(simplex[i])
                simplex_f[i] = fx
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()

            gamma = 1.0
            rho = 2.0
            sigma = 0.5
            eps = 0.5

            no_improve = 0
            max_no_improve = 20

            evals_in_this_run = n_simplex

            while evals_in_this_run < max_evals and calls_made[0] < budget and time.time() < deadline:
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                centroid = simplex[1:].mean(axis=0)
                if np.max(np.abs(simplex_f - simplex_f[0])) < 1e-10:
                    break

                xr = centroid + gamma * (centroid - simplex[0])
                xr = np.clip(xr, -5, 5)
                fxr = counted_func(xr)
                evals_in_this_run += 1

                if fxr < simplex_f[0]:
                    xe = centroid + rho * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fxe = counted_func(xe)
                    evals_in_this_run += 1
                    if fxe < fxr:
                        simplex[0] = xe
                        simplex_f[0] = fxe
                    else:
                        simplex[0] = xr
                        simplex_f[0] = fxr
                    no_improve = 0
                elif fxr < simplex_f[-1]:
                    simplex[0] = xr
                    simplex_f[0] = fxr
                    no_improve = 0
                else:
                    if fxr < simplex_f[0]:
                        xc = centroid + sigma * (xr - centroid)
                    else:
                        xc = centroid - sigma * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = counted_func(xc)
                    evals_in_this_run += 1

                    if fxc <= min(fxr, simplex_f[0]):
                        simplex[0] = xc
                        simplex_f[0] = fxc
                        no_improve = 0
                    else:
                        for i in range(1, n_simplex):
                            if calls_made[0] >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + eps * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = counted_func(simplex[i])
                            evals_in_this_run += 1
                        no_improve += 1

                local_min = simplex_f.min()
                if local_min < best_f:
                    best_f = local_min
                    idx_min = np.argmin(simplex_f)
                    best_x = simplex[idx_min].copy()

                if no_improve >= max_no_improve:
                    no_improve = 0
                    max_no_improve = min(max_no_improve * 2, 50)
                    worst_idx = np.argmax(simplex_f)
                    simplex[worst_idx] = simplex[0] + rng.normal(0, 0.5, dim_local)
                    simplex[worst_idx] = np.clip(simplex[worst_idx], -5, 5)
                    simplex_f[worst_idx] = counted_func(simplex[worst_idx])
                    evals_in_this_run += 1

            return simplex_f.min()

        for i in range(n_starts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            start_x = base_starts[i].copy()
            start_x = np.clip(start_x, -5, 5)

            if i > 0 and best_f < -50:
                budget_i = max(10, budgets_per_start[i] // 2)
            else:
                budget_i = budgets_per_start[i]

            nelder_mead(start_x, budget_i)

        if calls_made[0] < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.2
            for _ in range(40):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        fx_trial = counted_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.005:
                        break
                else:
                    step = max(0.03, step * 0.85)

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f
