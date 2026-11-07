"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track best value and best point
        best_x = None
        best_f = np.inf

        # We'll do multiple Nelder-Mead runs
        # Allocate budget: first run gets more, subsequent runs get less
        # Strategy: 5 runs with decreasing budgets
        # Run 1: 200 calls, Run 2: 100, Run 3: 80, Run 4: 60, Run 5: 40, rest: 20
        run_budgets = [200, 100, 80, 60, 40]
        remaining = budget - sum(run_budgets)
        if remaining > 0:
            run_budgets.append(remaining)

        calls_used = 0

        for run_idx, rb in enumerate(run_budgets):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize simplex: diverse starting point
            if run_idx == 0:
                # First run: try a few initial points, pick the best
                init_candidates = []
                for _ in range(min(5, rb // 10 + 1)):
                    x0 = rng.uniform(-5, 5, dim)
                    fx0 = func(x0)
                    calls_used += 1
                    if fx0 < best_f:
                        best_f, best_x = fx0, x0.copy()
                    init_candidates.append((x0, fx0))
                if not init_candidates:
                    continue
                # Pick the best initial point
                init_candidates.sort(key=lambda t: t[1])
                x0, f0 = init_candidates[0]
                if f0 < best_f:
                    best_f, best_x = f0, x0.copy()
            else:
                # Subsequent runs: random new start, but also consider best so far
                if rng.random() < 0.3 and best_x is not None:
                    # Start near the current best with some perturbation
                    x0 = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                f0 = func(x0)
                calls_used += 1
                if f0 < best_f:
                    best_f, best_x = f0, x0.copy()

            # Build simplex around x0
            # Standard Nelder-Mead: dim+1 vertices
            # Vertex 0: x0, others: x0 + delta * e_i
            delta = 0.5
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(dim):
                vi = x0.copy()
                vi[i] += delta
                vi = np.clip(vi, -5, 5)
                fi = func(vi)
                calls_used += 1
                if fi < best_f:
                    best_f, best_x = fi, vi.copy()
                simplex.append(vi)
                fvals.append(fi)

            # Nelder-Mead iteration
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            max_iter = rb - (dim + 1)  # remaining budget for this run

            for it in range(max_iter):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort vertices by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f, best_x = f_r, x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f, best_x = f_e, x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < best_f:
                        best_f, best_x = f_c, x_c.copy()

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = func(simplex[i])
                            calls_used += 1
                            if f_i < best_f:
                                best_f, best_x = f_i, simplex[i].copy()
                            fvals[i] = f_i
                        if calls_used >= budget:
                            break
                        if time.time() > deadline:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
