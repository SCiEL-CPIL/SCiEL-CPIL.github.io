"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive simplex for Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        def nelder_mead(x_init, budget):
            """Nelder-Mead with adaptive step size."""
            x = x_init.copy()
            fx = func(x)
            calls = 1

            # Create initial simplex
            simplex = [(x.copy(), fx)]
            base_step = 0.5

            for i in range(dim):
                xi = x.copy()
                xi[i] += base_step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                if time.time() > deadline or calls >= total_budget:
                    break
                simplex.append((xi, fi))

            # Ensure we have dim+1 points
            while len(simplex) < dim + 1 and calls < total_budget and time.time() < deadline:
                xi = x + base_step * rng.standard_normal(dim)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append((xi, fi))

            if len(simplex) < 2:
                return x, fx, calls

            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5  # Shrink

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                # Sort simplex by function value
                simplex.sort(key=lambda p: p[1])

                # Get best and worst
                best = simplex[0]
                worst = simplex[-1]

                # Check convergence
                if len(simplex) >= 2:
                    if abs(best[1] - worst[1]) < 1e-12:
                        break

                # Compute centroid of all points except worst
                centroid = np.zeros(dim)
                for i in range(len(simplex) - 1):
                    centroid += simplex[i][0]
                centroid /= (len(simplex) - 1)

                # Reflection
                xr = centroid + alpha * (centroid - worst[0])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < best[1]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst[0])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = (xe, fe)
                    else:
                        simplex[-1] = (xr, fr)
                elif fr < worst[1]:
                    # Reflection is good
                    simplex[-1] = (xr, fr)
                else:
                    # Contraction
                    if fr < worst[1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst[0] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < worst[1]:
                        simplex[-1] = (xc, fc)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i][0] = best[0] + sigma_nm * (simplex[i][0] - best[0])
                            simplex[i][0] = np.clip(simplex[i][0], -5, 5)
                            simplex[i][1] = func(simplex[i][0])
                            calls += 1
                            if time.time() > deadline or calls >= total_budget:
                                break
                        if time.time() > deadline or calls >= total_budget:
                            break

            # Return best point
            simplex.sort(key=lambda p: p[1])
            return simplex[0][0], simplex[0][1], calls

        def run_es(x_init, budget):
            """(1+1)-ES with success rule for exploration."""
            x = x_init.copy()
            fx = func(x)
            calls = 1
            sigma = 2.0
            min_sigma = 1e-4
            max_sigma = 5.0

            # 1/5 success rule parameters
            window = 10
            successes = 0
            c_inc = 1.1
            c_dec = 0.9

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                else:
                    successes += 0

                # Adapt sigma every 'window' steps
                if successes >= window:
                    sigma = min(sigma * c_inc, max_sigma)
                    successes = 0
                elif successes == 0:
                    # Check if we've had 0 successes in a full window
                    pass

            return x, fx, calls

        # Strategy: Multi-start Nelder-Mead with ES exploration
        # Phase 1: Quick ES exploration from multiple starts to find good basins
        # Phase 2: Nelder-Mead refinement on the best points

        if total_budget < 20:
            x_init = rng.uniform(-5, 5, dim)
            x_final, f_final, _ = run_es(x_init, total_budget)
            return x_final, f_final

        # Use 40% of budget for ES exploration with 4 starts
        n_explore = 4
        budget_explore_each = max(10, total_budget // (n_explore + 2))
        budget_nm_each = max(20, (total_budget - n_explore * budget_explore_each) // n_explore)

        best_x = None
        best_f = float('inf')
        total_calls = 0
        candidates = []

        for i in range(n_explore):
            if time.time() > deadline:
                break
            if total_calls >= total_budget:
                break

            x_init = rng.uniform(-5, 5, dim)
            x_run, f_run, calls_used = run_es(x_init, budget_explore_each)
            total_calls += calls_used

            candidates.append((x_run, f_run))
            if f_run < best_f:
                best_f = f_run
                best_x = x_run

        # Phase 2: Nelder-Mead on top candidates
        candidates.sort(key=lambda t: t[1])
        n_refine = min(n_explore, len(candidates))

        for i in range(n_refine):
            if time.time() > deadline:
                break
            if total_calls >= total_budget:
                break

            x_init = candidates[i][0]
            remaining_budget = total_budget - total_calls
            x_final, f_final, calls_used = nelder_mead(x_init, min(budget_nm_each, remaining_budget))
            total_calls += calls_used

            if f_final < best_f:
                best_f = f_final
                best_x = x_final

        return best_x, best_f
# EVOLVE-BLOCK-END
