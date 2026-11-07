"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 270.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f, calls_used
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, alpha=1.0, gamma=2.0, rho=0.5, max_calls=200):
            nonlocal calls_used, best_x, best_f
            local_calls = 0
            x0 = np.clip(x0, -5, 5)

            # Initialize simplex
            simplex = []
            fvals = []

            f0 = func(x0)
            calls_used += 1
            local_calls += 1
            simplex.append(x0.copy())
            fvals.append(f0)
            update_best(x0, f0)

            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] + 0.5 <= 5 else -0.5
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_used += 1
                local_calls += 1
                simplex.append(xi.copy())
                fvals.append(fi)
                update_best(xi, fi)

            simplex = np.array(simplex)
            fvals = np.array(fvals)

            no_improve = 0
            prev_best = fvals.copy().min()
            max_no_improve = 30

            while local_calls < max_calls:
                if calls_used >= budget or time.time() > deadline - 10.0:
                    break

                # Sort simplex by fitness
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                local_calls += 1
                update_best(x_ref, f_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_used += 1
                    local_calls += 1
                    update_best(x_exp, f_exp)

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Just reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    local_calls += 1
                    update_best(x_con, f_con)

                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink toward best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shrink = func(simplex[i])
                            calls_used += 1
                            local_calls += 1
                            fvals[i] = f_shrink
                            update_best(simplex[i], f_shrink)
                            if calls_used >= budget or time.time() > deadline - 10.0:
                                break

                # Check improvement
                current_best = fvals.min()
                if current_best < prev_best - 1e-10:
                    no_improve = 0
                    prev_best = current_best
                else:
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

            return local_calls

        # Multi-start Nelder-Mead
        starts = [np.zeros(dim)]
        for _ in range(3):
            starts.append(rng.uniform(-3, 3, size=dim))

        # Budget allocation: give more to early starts for early progress
        nm_budgets = [180, 100, 80, 60]

        for i, sx in enumerate(starts):
            if calls_used >= budget or time.time() > deadline - 20.0:
                break
            nelder_mead(sx, max_calls=nm_budgets[i])

        # Phase 2: (1+1)-ES polish
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            max_polish = min(50, budget - calls_used)
            for _ in range(max_polish):
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.06
                    update_best(x, fx)
                else:
                    sigma *= 0.82

            # Fine refinement
            sigma = max(sigma, 0.02)
            max_fine = min(30, budget - calls_used)
            for _ in range(max_fine):
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                    update_best(x, fx)
                else:
                    sigma *= 0.88

            # Final micro-tuning with remaining budget
            if calls_used < budget:
                x = best_x.copy()
                fx = best_f
                sigma = 0.005
                while calls_used < budget and time.time() < deadline - 2.0:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.03
                        update_best(x, fx)
                    else:
                        sigma *= 0.75
                        if sigma < 1e-7:
                            break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
