"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with box constraints and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time() > 1.0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Nelder-Mead simplex parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        best_x = None
        best_f = np.inf
        calls_used = 0

        def nm_search(x0, f0, max_iter, initial_sigma):
            """Run Nelder-Mead from x0 with f0 already evaluated."""
            nonlocal calls_used
            n = dim
            # Build initial simplex
            simplex = [x0.copy()]
            simplexf = [f0]
            # Generate other points by moving along each axis
            for i in range(n):
                xi = x0.copy()
                step = initial_sigma if x0[i] + initial_sigma <= 5.0 else -initial_sigma
                if x0[i] + step < -5.0:
                    step = -step
                xi[i] += step
                xi = clip(xi)
                fi = func(xi)
                calls_used += 1
                simplex.append(xi)
                simplexf.append(fi)

            # Sort simplex by function value
            def sort_simplex():
                idx = np.argsort(simplexf)
                for i in range(n + 1):
                    simplex[i] = simplex[idx[i]].copy()
                    simplexf[i] = simplexf[idx[i]]

            sort_simplex()

            for _ in range(max_iter):
                if not time_left():
                    break

                # Compute centroid excluding worst point
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                xr = clip(centroid + alpha * (centroid - simplex[n]))
                fr = func(xr)
                calls_used += 1

                if fr < simplexf[0]:
                    # Expansion
                    xe = clip(centroid + gamma * (centroid - simplex[n]))
                    fe = func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[n] = xe
                        simplexf[n] = fe
                    else:
                        simplex[n] = xr
                        simplexf[n] = fr
                elif fr < simplexf[n - 1]:
                    # Accept reflection
                    simplex[n] = xr
                    simplexf[n] = fr
                else:
                    # Contraction
                    if fr < simplexf[n]:
                        # Outside contraction
                        xc = clip(centroid + rho * (centroid - simplex[n]))
                        fc = func(xc)
                        calls_used += 1
                        if fc <= fr:
                            simplex[n] = xc
                            simplexf[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                                fi = func(simplex[i])
                                calls_used += 1
                                simplexf[i] = fi
                    else:
                        # Inside contraction
                        xc = clip(centroid - rho * (centroid - simplex[n]))
                        fc = func(xc)
                        calls_used += 1
                        if fc < simplexf[n]:
                            simplex[n] = xc
                            simplexf[n] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                                fi = func(simplex[i])
                                calls_used += 1
                                simplexf[i] = fi

                sort_simplex()

                # Check convergence
                if n > 0:
                    f_range = simplexf[n] - simplexf[0]
                    if f_range < 1e-12 * (1.0 + abs(simplexf[0])):
                        break

            return simplex[0], simplexf[0]

        # Initial point: center of box
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_used += 1

        if not time_left():
            return x0, f0

        # First NM run with moderate initial spread
        sigma_init = 1.0
        max_iter_1 = min(budget - calls_used, 400)
        x_best, f_best = nm_search(x0, f0, max_iter_1, sigma_init)

        if f_best < best_f:
            best_f = f_best
            best_x = x_best.copy()

        # If budget remains, do a second NM run with smaller spread for polish
        if time_left() and calls_used < budget:
            remaining = budget - calls_used
            if remaining > 20:
                max_iter_2 = min(remaining, 150)
                sigma_init_2 = 0.3
                x2, f2 = nm_search(x_best, best_f, max_iter_2, sigma_init_2)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2.copy()

        # Final polish: coordinate descent with small steps
        if time_left() and calls_used < budget:
            remaining = budget - calls_used
            if remaining > 10:
                x = best_x.copy()
                f = best_f
                step = 0.05
                for _ in range(min(remaining, 50)):
                    if not time_left():
                        break
                    improved = False
                    for i in range(dim):
                        for direction in [-1, 1]:
                            xn = x.copy()
                            xn[i] += direction * step
                            xn = clip(xn)
                            fn = func(xn)
                            calls_used += 1
                            if fn < f:
                                x, f = xn, fn
                                improved = True
                        if not time_left():
                            break
                    if not improved:
                        step *= 0.5
                        if step < 1e-6:
                            break
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
