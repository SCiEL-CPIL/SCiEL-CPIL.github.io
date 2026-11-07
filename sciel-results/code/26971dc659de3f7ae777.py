"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Nelder-Mead simplex-based optimizer with adaptive step size and
    multiple restarts, optimized for AOCC (early progress matters).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        def check_time():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Quick initial exploration - find a reasonable starting point
        n_init = min(15, total_budget)
        for i in range(n_init):
            if check_time() or calls_made >= total_budget:
                break
            x = rng.uniform(-5, 5, n)
            eval_point(x)

        if best_x is None:
            x0 = np.zeros(n)
            best_f = eval_point(x0)
            best_x = x0.copy()

        # Phase 2: Nelder-Mead with restarts
        # Nelder-Mead uses n+1 points per iteration, so budget_per_restart is in iterations
        max_restarts = 6
        remaining = total_budget - calls_made
        # Each NM iteration uses n+1 evaluations (but we reuse some), so roughly n+1 per iter
        budget_per_restart = max(5, remaining // (max_restarts * (n + 1)))

        for restart in range(max_restarts):
            if calls_made >= total_budget or check_time():
                break

            # Initialize simplex
            if restart == 0:
                x_center = best_x.copy()
                step = 0.5
            else:
                x_center = best_x + rng.normal(0, 0.3, n)
                x_center = np.clip(x_center, -5, 5)
                step = 0.3 + 0.3 * rng.random()

            # Build simplex
            simplex = [x_center.copy()]
            fvals = [best_f]
            for i in range(n):
                x_i = x_center.copy()
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                f_i = eval_point(x_i)
                simplex.append(x_i)
                fvals.append(f_i)

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrinkage
            max_iter = budget_per_restart

            for it in range(max_iter):
                if calls_made >= total_budget or check_time():
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check if converged
                if len(fvals) >= 2 and abs(fvals[0] - fvals[-1]) < 1e-10:
                    break

                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = eval_point(x_reflect)

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = eval_point(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Just accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = eval_point(x_contract)
                    if f_contract < fvals[-1]:
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_point(simplex[i])

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[order[0]].copy()

        # Final polish: local search around the best point
        if calls_made < total_budget and not check_time():
            local_sigma = 0.1
            for _ in range(min(50, total_budget - calls_made)):
                if calls_made >= total_budget or check_time():
                    break
                x_try = best_x + local_sigma * rng.standard_normal(n)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls_made += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    local_sigma *= 0.9
                else:
                    local_sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
