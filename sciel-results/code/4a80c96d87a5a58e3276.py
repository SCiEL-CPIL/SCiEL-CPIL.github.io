"""Rank-1 + Rank-mu diagonal adaptation for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal CMA-ES variant with rank-1 + rank-mu updates for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        n_calls = 0

        def timed_out():
            return time.time() > deadline

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or timed_out():
                return None
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(y)
            n_calls += 1
            if f_val < best_f:
                best_f = float(f_val)
                best_x = y.copy()
            return f_val

        def nelder_mead_polish(x0, max_calls, init_step=0.1):
            """Nelder-Mead polish with budget tracking."""
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or timed_out():
                return

            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += init_step

            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return
                fvals[i] = f

            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

            sort_simplex()

            for _ in range(max_calls):
                if n_calls >= budget or timed_out():
                    break
                if np.std(fvals) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break

                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    gamma = 2.0
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        sigma = 0.5
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return
                            fvals[i] = f
                        sort_simplex()

        # Diagonal adaptation parameters
        mean = np.zeros(dim)
        sigma = 2.0  # initial step size to cover the box
        sigma_sq = sigma ** 2  # per-dimension variance (diagonal)
        # We track a single sigma for simplicity, but adapt per-dimension via sigma_sq

        # Population size schedule
        def get_pop_size(remaining_calls):
            # Start with larger population, shrink as budget depletes
            if remaining_calls > 200:
                return 20
            elif remaining_calls > 100:
                return 15
            elif remaining_calls > 50:
                return 10
            else:
                return 8

        # Main loop
        while n_calls < budget and not timed_out():
            remaining = budget - n_calls
            if remaining < 5:
                break

            lam = get_pop_size(remaining)
            mu = max(1, lam // 2)

            # Sample population
            samples = np.zeros((lam, dim))
            fvals = np.zeros(lam)
            valid = 0
            for i in range(lam):
                # Use per-dimension sigma
                x = mean + sigma * rng.standard_normal(dim)
                x = np.clip(x, -5.0, 5.0)
                f = call_f(x)
                if f is None:
                    break
                samples[i] = x
                fvals[i] = f
                valid += 1
            if valid == 0:
                break
            lam = valid
            if lam < 2:
                break
            mu = min(mu, lam)

            # Sort by fitness
            idx = np.argsort(fvals)
            samples = samples[idx]
            fvals = fvals[idx]

            # Update mean: weighted combination of top mu
            weights = np.zeros(mu)
            weights[0] = np.log(mu) - np.sum(np.log(np.arange(1, mu + 1)))
            weights = np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mean = np.sum(samples[:mu] * weights[:, None], axis=0)

            # Rank-1 update: best point
            y_best = (samples[0] - mean) / sigma
            c1 = 2 / ((dim + 1) ** 2 + 2)

            # Rank-mu update: top mu points
            ys = (samples[:mu] - mean) / sigma
            cmu = min(0.5 - c1 / 2, (2 * mu - 2) / ((dim + 2) ** 2 + 2))
            cmu = max(cmu, 0.0)

            # Update sigma_sq (diagonal variance)
            # For each dimension, compute the natural gradient step
            # sigma_sq[d] *= (1 + c1 * (y_best[d]^2 - 1) + cmu * sum(w_i * (ys[i,d]^2 - 1)))
            # But we use a single sigma, so we take the geometric mean update
            # Actually, for diagonal adaptation, we track per-dimension sigma
            # Let's switch to per-dimension sigma

            # Per-dimension sigma update
            # sigma_sq[d] *= exp(c1 * (y_best[d]^2 - 1) + cmu * sum_i w_i * (ys[i,d]^2 - 1))
            # This is the natural gradient for the log-variance
            log_sigma_sq_update = c1 * (y_best ** 2 - 1)
            for i in range(mu):
                log_sigma_sq_update += cmu * weights[i] * (ys[i] ** 2 - 1)
            # Damping: use a fraction of the update
            damping = 0.5
            sigma_sq = sigma_sq * np.exp(damping * log_sigma_sq_update)
            # Clip sigma_sq to avoid explosion or collapse
            sigma_sq = np.clip(sigma_sq, 1e-6, 100.0)
            sigma = np.sqrt(sigma_sq)

            # If sigma becomes too small, we've converged
            if np.max(sigma) < 1e-4:
                break

        # Polish phase: use remaining budget for Nelder-Mead on best point
        if n_calls < budget and not timed_out():
            remaining = budget - n_calls
            if remaining >= 10:
                # Adaptive initial step based on current sigma
                init_step = max(0.01, np.mean(sigma) * 0.5)
                nelder_mead_polish(best_x, max_calls=min(remaining, 80), init_step=init_step)

        return best_x, best_f
# EVOLVE-BLOCK-END
