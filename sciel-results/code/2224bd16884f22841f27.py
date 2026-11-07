"""Ridge-aware optimizer: multi-start Nelder-Mead with ridge detection and line search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with ridge-aware polishing for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        def evaluate(x):
            nonlocal calls_used
            x = clip(x)
            f = func(x)
            calls_used += 1
            update_best(x, f)
            return x, f

        def nelder_mead(x0, step_size, max_iter=100):
            """Run Nelder-Mead from x0 with given initial step size."""
            nonlocal calls_used, best_x, best_f

            if not time_ok() or calls_used >= budget - 10:
                return

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            # Build simplex
            simplex = [x0.copy()]
            simplex_f = [best_f]
            # Re-evaluate x0 to get consistent value
            x0_eval, f0 = evaluate(x0)
            simplex[0] = x0_eval
            simplex_f[0] = f0

            for i in range(dim):
                if not time_ok() or calls_used >= budget - 10:
                    return
                xi = x0_eval.copy()
                xi[i] += step_size
                xi_eval, fi = evaluate(xi)
                simplex.append(xi_eval)
                simplex_f.append(fi)

            # Main loop
            for _ in range(max_iter):
                if not time_ok() or calls_used >= budget - 10:
                    break

                # Sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Convergence check
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r_eval, f_r = evaluate(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    if not time_ok() or calls_used >= budget - 10:
                        break
                    x_e = centroid + gamma * (x_r_eval - centroid)
                    x_e_eval, f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e_eval
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r_eval
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r_eval
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if not time_ok() or calls_used >= budget - 10:
                        break
                    if f_r < worst_f:
                        x_c = centroid + rho * (x_r_eval - centroid)
                    else:
                        x_c = centroid - rho * (centroid - worst)
                    x_c_eval, f_c = evaluate(x_c)

                    if f_c < min(f_r, worst_f):
                        simplex[-1] = x_c_eval
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok() or calls_used >= budget - 10:
                                return
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i], simplex_f[i] = evaluate(simplex[i])

        def ridge_line_search(x_center, direction, max_calls=15):
            """1D line search along direction from x_center."""
            nonlocal calls_used, best_x, best_f

            if not time_ok() or calls_used >= budget - 5:
                return x_center

            # First try a few steps to find a good direction
            t_best = 0.0
            x_best = x_center.copy()
            f_best = best_f

            for t in [0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 0.2, 5.0, -5.0]:
                if not time_ok() or calls_used >= budget - 5:
                    break
                x_try = clip(x_center + t * direction)
                f_try = func(x_try)
                calls_used += 1
                if f_try < f_best:
                    f_best = f_try
                    t_best = t
                    x_best = x_try.copy()
                    update_best(x_best, f_best)

            # If we found improvement, do a finer search around t_best
            if t_best != 0.0:
                for t in [t_best * 0.5, t_best * 1.5, t_best * 0.75, t_best * 1.25]:
                    if not time_ok() or calls_used >= budget - 5:
                        break
                    x_try = clip(x_center + t * direction)
                    f_try = func(x_try)
                    calls_used += 1
                    if f_try < f_best:
                        f_best = f_try
                        x_best = x_try.copy()
                        update_best(x_best, f_best)

            return x_best

        def detect_ridge_direction(x_center, step=0.5, max_calls=20):
            """Try to detect the ridge direction by probing."""
            nonlocal calls_used

            if not time_ok() or calls_used >= budget - 10:
                return None

            # Probe along each axis
            axis_improvements = []
            for i in range(dim):
                if not time_ok() or calls_used >= budget - 10:
                    break
                for sign in [1.0, -1.0]:
                    x_try = clip(x_center + sign * step * np.eye(dim)[:, i])
                    f_try = func(x_try)
                    calls_used += 1
                    if f_try < best_f - 1e-10:
                        axis_improvements.append((i, sign, f_try))

            if not axis_improvements:
                return None

            # If multiple axes improve, try combinations
            # For ridge, typically one or two dominant directions
            best_axis_impr = min(axis_improvements, key=lambda x: x[2])
            i_best, sign_best, _ = best_axis_impr
            direction = np.zeros(dim)
            direction[i_best] = sign_best
            return direction

        def polish(x_center, max_polish_calls=40):
            """Polish using line search in detected directions and coordinate descent."""
            nonlocal calls_used, best_x, best_f

            x = x_center.copy()
            f_curr = best_f

            # Try to detect ridge direction
            if not time_ok() or calls_used >= budget - 10:
                return

            direction = detect_ridge_direction(x, step=0.5, max_calls=15)

            if direction is not None:
                x = ridge_line_search(x, direction, max_calls=10)

            # Coordinate descent polish
            for _ in range(5):
                if not time_ok() or calls_used >= budget - 5:
                    break
                improved = False
                for i in range(dim):
                    if not time_ok() or calls_used >= budget - 5:
                        break
                    # Try small steps
                    for step in [0.1, 0.05, 0.2]:
                        for sign in [1.0, -1.0]:
                            if not time_ok() or calls_used >= budget - 5:
                                break
                            x_try = clip(x + sign * step * np.eye(dim)[:, i])
                            f_try = func(x_try)
                            calls_used += 1
                            if f_try < f_curr:
                                x = x_try
                                f_curr = f_try
                                improved = True
                                update_best(x, f_curr)
                                break
                if not improved:
                    break

        # === Main search ===

        # Phase 1: Initial exploration with diverse starts
        starts = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
        ]

        for idx, start in enumerate(starts):
            if not time_ok() or calls_used >= budget - 30:
                break
            step = 2.0 if idx == 0 else 1.5
            nelder_mead(start, step, max_iter=60)

        # Phase 2: More focused NM from best and nearby
        if best_x is not None and time_ok() and calls_used < budget - 30:
            # NM from best with smaller step
            nelder_mead(best_x, 0.5, max_iter=40)

            # NM from perturbed best
            if time_ok() and calls_used < budget - 20:
                perturbed = best_x + rng.normal(0, 0.5, dim)
                nelder_mead(perturbed, 0.5, max_iter=30)

        # Phase 3: Polish
        if best_x is not None and time_ok() and calls_used < budget - 10:
            polish(best_x, max_polish_calls=40)

        # Phase 4: If budget remains, try a few more random restarts
        remaining = budget - calls_used - 10
        if remaining > 30 and time_ok():
            for _ in range(min(3, remaining // 30)):
                if not time_ok() or calls_used >= budget - 10:
                    break
                rand_start = best_x + rng.normal(0, 1.0, dim) if best_x is not None else rng.uniform(-4, 4, dim)
                nelder_mead(rand_start, 1.0, max_iter=20)
                if best_x is not None and time_ok() and calls_used < budget - 10:
                    polish(best_x, max_polish_calls=10)

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
