"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Nelder-Mead local refinement and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Nelder-Mead helper
        def nelder_mead(func, x0, budget, deadline, dim, rng):
            nonlocal calls
            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(np.clip(xi, -5, 5))

            fx_values = []
            for i in range(dim + 1):
                if calls >= budget or time.time() >= deadline:
                    break
                fx_values.append(func(simplex[i]))
                calls += 1

            if len(fx_values) < dim + 1:
                return x0, fx_values[0] if fx_values else fx

            # Main NM loop
            max_nm_iter = 200
            for _ in range(max_nm_iter):
                if calls >= budget or time.time() >= deadline:
                    break

                # Sort by function value
                order = np.argsort(fx_values)
                simplex = [simplex[i].copy() for i in order]
                fx_values = [fx_values[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time.time() >= deadline:
                    break
                fxr = func(xr)
                calls += 1

                if fxr < fx_values[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls < budget and time.time() < deadline:
                        fxe = func(xe)
                        calls += 1
                        if fxe < fxr:
                            simplex[-1] = xe
                            fx_values[-1] = fxe
                        else:
                            simplex[-1] = xr
                            fx_values[-1] = fxr
                    else:
                        simplex[-1] = xr
                        fx_values[-1] = fxr
                elif fxr < fx_values[-2]:
                    simplex[-1] = xr
                    fx_values[-1] = fxr
                else:
                    # Contraction
                    if fxr < fx_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls < budget and time.time() < deadline:
                        fxc = func(xc)
                        calls += 1
                        if fxc <= fxr if fxr < fx_values[-1] else fxc <= fx_values[-1]:
                            simplex[-1] = xc
                            fx_values[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls < budget and time.time() < deadline:
                                    fx_values[i] = func(simplex[i])
                                    calls += 1
                            break
                    else:
                        break

            # Return best from simplex
            best_idx = np.argmin(fx_values)
            return simplex[best_idx].copy(), fx_values[best_idx]

        sigma = 2.0
        es_step = 0

        while calls < self.budget and time.time() < deadline:
            # Every 30 ES steps, try Nelder-Mead refinement
            if es_step % 30 == 0 and es_step > 0:
                nm_budget = min(150, self.budget - calls)
                if nm_budget > 10:
                    nm_x, nm_fx = nelder_mead(func, x, self.budget, deadline, dim, rng)
                    if nm_fx < fx:
                        x, fx = nm_x, nm_fx
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                    es_step = 0  # Reset counter after NM
                    continue

            # ES step
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            es_step += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

            # Final NM refinement if budget is low
            if self.budget - calls < 50:
                nm_budget = min(40, self.budget - calls)
                if nm_budget > 5:
                    nm_x, nm_fx = nelder_mead(func, x, self.budget, deadline, dim, rng)
                    if nm_fx < fx:
                        x, fx = nm_x, nm_fx
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
