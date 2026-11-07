"""Optimizer for BBOB-style f17 (Schaffers F7) using Nelder-Mead with adaptive restarts."""
import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, f0, delta_init, max_calls, max_iter=200):
            nonlocal calls_used, best_x, best_f

            # Nelder-Mead parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrinkage

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                step = delta_init if x0[i] + delta_init <= 5 else -delta_init
                if x0[i] + step > 5:
                    step = -delta_init
                xi[i] += step
                simplex.append(xi)

            fs = [f0]
            for i in range(1, dim + 1):
                fi = func(simplex[i])
                calls_used += 1
                fs.append(fi)
                update_best(simplex[i], fi)

            iter_count = 0
            no_improve = 0

            while iter_count < max_iter and calls_used < max_calls:
                if time.time() > deadline - 3.0:
                    break

                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Check convergence
                if fs[0] < best_f - 1e-12:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 15:
                        break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]

                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                update_best(xr, fr)

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    update_best(xe, fe)
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    update_best(xc, fc)
                    if fc < fr:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_used += 1
                            update_best(simplex[i], fi)
                            fs[i] = fi

                iter_count += 1

            return calls_used

        # Phase 1: Initial exploration with multiple random starts
        # Try a few different starting points to handle the hidden shift
        n_starts = 5
        delta_init = 1.0

        for start_idx in range(n_starts):
            if calls_used >= budget:
                break
            if time.time() > deadline - 10.0:
                break

            if start_idx == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, size=dim)

            f0 = func(x0)
            calls_used += 1
            update_best(x0, f0)

            # Run Nelder-Mead from this start
            remaining = budget - calls_used
            if remaining < 10:
                break

            nelder_mead(x0, f0, delta_init, calls_used + remaining)

        # Phase 2: Fine-tuning with (1+1)-ES
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(30):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    update_best(x, fx)
                else:
                    sigma *= 0.85

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
