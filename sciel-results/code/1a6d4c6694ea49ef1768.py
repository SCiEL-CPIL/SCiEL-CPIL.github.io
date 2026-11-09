"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Metropolis acceptance, coordinate descent, and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(np.asarray(x, dtype=float), -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, step=0.5, max_iter=80, use_metropolis=False, T=1.0):
            nonlocal calls, best_x, best_f, deadline

            n = self.dim
            # Build simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0.copy()
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step

            simplex_f = np.array([eval_func(simplex[i]) for i in range(n + 1)])

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            rng = np.random.default_rng()

            # Track best in this NM run for Metropolis
            local_best_f = simplex_f.min()
            no_improve = 0
            max_no_improve = 12

            for it in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                # Convergence check
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-13 * (1 + abs(simplex_f[0])):
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -self.bound, self.bound)
                f_r = eval_func(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -self.bound, self.bound)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -self.bound, self.bound)
                    f_c = eval_func(x_c)

                    if f_c <= max(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -self.bound, self.bound)
                            simplex_f[i] = eval_func(simplex[i])

                # Track improvement
                current_best = simplex_f.min()
                if current_best < local_best_f - 1e-12:
                    local_best_f = current_best
                    no_improve = 0
                else:
                    no_improve += 1

                # Metropolis-like perturbation to escape local minima
                if use_metropolis and no_improve >= max_no_improve:
                    # Perturb the best point with a random kick
                    kick = rng.standard_normal(n) * step * 0.5
                    x_kick = simplex[0] + kick
                    x_kick = np.clip(x_kick, -self.bound, self.bound)
                    f_kick = eval_func(x_kick)

                    # Metropolis acceptance: accept if better, or with probability exp(-delta/T)
                    delta = f_kick - simplex_f[0]
                    if f_kick < simplex_f[0] or (delta < 50.0 and rng.random() < np.exp(-delta / max(T, 1e-6))):
                        simplex[-1] = x_kick
                        simplex_f[-1] = f_kick
                        no_improve = 0
                        step *= 0.7
                    else:
                        # Try a larger kick
                        kick = rng.standard_normal(n) * step
                        x_kick = simplex[0] + kick
                        x_kick = np.clip(x_kick, -self.bound, self.bound)
                        f_kick = eval_func(x_kick)
                        if f_kick < simplex_f[0]:
                            simplex[-1] = x_kick
                            simplex_f[-1] = f_kick
                            no_improve = 0
                        step *= 0.5

                # Adaptive step size: shrink if simplex is small, grow if stuck
                simplex_size = np.max(np.abs(simplex - simplex[0]), axis=0)
                if np.mean(simplex_size) < step * 0.05:
                    break
                if no_improve > 8:
                    step *= 0.7

            return simplex[0], local_best_f

        def coordinate_descent(x0, max_iter=30, initial_step=0.5):
            """Coordinate descent polish with adaptive step size."""
            nonlocal calls, best_x, best_f, deadline

            n = self.dim
            x = x0.copy()
            f = eval_func(x)
            step = initial_step

            for it in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    # Try positive direction
                    x_plus = x.copy()
                    x_plus[i] += step
                    x_plus = np.clip(x_plus, -self.bound, self.bound)
                    f_plus = eval_func(x_plus)

                    # Try negative direction
                    x_minus = x.copy()
                    x_minus[i] -= step
                    x_minus = np.clip(x_minus, -self.bound, self.bound)
                    f_minus = eval_func(x_minus)

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        improved = True
                    elif f_minus < f:
                        x = x_minus
                        f = f_minus
                        improved = True
                    else:
                        # Shrink step for this coordinate
                        step *= 0.5

                if not improved:
                    break
                if step < 1e-6:
                    break

            return x, f

        # Main search: multi-start Nelder-Mead
        rng = np.random.default_rng(123)

        # Initial evaluations: spread out to find good basins
        init_points = [np.zeros(self.dim)]
        # Axis-aligned points at multiple scales
        for i in range(self.dim):
            for scale in [2.0, -2.0]:
                p = np.zeros(self.dim)
                p[i] = scale
                init_points.append(p.copy())
        # Corner-ish points
        init_points.append(np.full(self.dim, 3.0))
        init_points.append(np.full(self.dim, -3.0))
        # Random points
        for _ in range(8):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))

        for p in init_points:
            if calls < self.budget:
                eval_func(p)

        # Multi-start Nelder-Mead
        # First run from best initial point
        if calls < self.budget:
            x0, _ = nelder_mead(best_x.copy(), step=1.0, max_iter=50, use_metropolis=True, T=2.0)

        # If we still have budget, do additional starts from random points
        remaining_budget = self.budget - calls
        if remaining_budget > 40 and time.time() < deadline:
            # Try 3-4 more random starts
            n_starts = min(4, remaining_budget // 35)
            for s in range(n_starts):
                if calls >= self.budget or time.time() > deadline:
                    break
                # Mix of local and global starts
                if s % 2 == 0:
                    x0 = best_x + rng.standard_normal(self.dim) * 1.0
                else:
                    x0 = rng.uniform(-4.5, 4.5, self.dim)
                x0 = np.clip(x0, -self.bound, self.bound)
                f_start = eval_func(x0)
                if calls < self.budget:
                    nelder_mead(x0, step=0.7, max_iter=35, use_metropolis=True, T=1.0)

        # Coordinate descent polish
        if calls < self.budget - 15 and time.time() < deadline:
            coordinate_descent(best_x.copy(), max_iter=25, initial_step=0.3)

        # Final polish with L-BFGS-B if we have budget left
        if calls < self.budget - 10 and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                x_start = best_x.copy()

                def wrapped_func(x):
                    nonlocal calls, best_x, best_f
                    if calls >= self.budget or time.time() > deadline:
                        return np.inf
                    x = np.clip(x, -self.bound, self.bound)
                    try:
                        f = float(func(x))
                    except Exception:
                        f = np.inf
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f

                res = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-self.bound, self.bound)] * self.dim,
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )

                if res.x is not None and calls < self.budget:
                    x_res = np.clip(res.x, -self.bound, self.bound)
                    eval_func(x_res)

            except Exception:
                pass

        # Second L-BFGS-B pass with tighter tolerance if budget allows
        if calls < self.budget - 5 and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                x_start = best_x.copy()

                def wrapped_func2(x):
                    nonlocal calls, best_x, best_f
                    if calls >= self.budget or time.time() > deadline:
                        return np.inf
                    x = np.clip(x, -self.bound, self.bound)
                    try:
                        f = float(func(x))
                    except Exception:
                        f = np.inf
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f

                res2 = minimize(
                    wrapped_func2,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-self.bound, self.bound)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 20}
                )

                if res2.x is not None and calls < self.budget:
                    x_res2 = np.clip(res2.x, -self.bound, self.bound)
                    eval_func(x_res2)

            except Exception:
                pass

        # One final short NM polish if budget remains
        if calls < self.budget - 3 and time.time() < deadline:
            nelder_mead(best_x.copy(), step=0.05, max_iter=15, use_metropolis=False)

        return best_x, best_f
# EVOLVE-BLOCK-END
