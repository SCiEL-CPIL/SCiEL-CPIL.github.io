"""Combined program: CMA-ES with Nelder-Mead polish for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin vs 300s/330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        best_x = np.zeros(dim)
        best_f = float('inf')
        n_calls = 0

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

        # ---- CMA-ES parameters for dim=5 ----
        lambda_ = 14
        mu_ = 7
        w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def nelder_mead(x0, max_calls, init_step=0.1):
            """Nelder-Mead with call budget tracking."""
            nonlocal n_calls, best_x, best_f

            if n_calls >= budget or timed_out():
                return

            # Initialize simplex
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
                    # Expansion
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
                    # Contraction
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
                        # Shrink
                        sigma = 0.5
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return
                            fvals[i] = f
                        sort_simplex()

        # ---- Phase 1: Diverse initial exploration to break Discus ring symmetry ----
        n_init = min(16, budget)
        for k in range(n_init):
            if timed_out() or n_calls >= budget:
                break
            if k == 0:
                x0 = np.zeros(dim)
            elif k == 1:
                x0 = 3.0 * rng.standard_normal(dim)
            elif k == 2:
                x0 = np.linspace(-4.0, 4.0, dim)
            else:
                x0 = 4.0 * rng.uniform(-1.0, 1.0, dim)
            call_f(x0)

        # ---- Phase 2: CMA-ES main loop ----
        # Start from the best point found so far
        x_mean = best_x.copy()
        sigma = 1.5
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        L = np.eye(dim)
        have_L = True

        # Track stagnation
        prev_best_f = best_f
        stagnation_count = 0
        max_stagnation = 20  # generations without improvement before restart

        while n_calls < budget and not timed_out():
            # Generate population
            offspring = []
            for _ in range(lambda_):
                if n_calls >= budget or timed_out():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                f = call_f(y)
                if f is None:
                    break
                offspring.append((f, y, z))

            if len(offspring) < mu_:
                break

            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu_]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
            delta = x_mean - old_mean

            # Evolution paths
            if have_L:
                try:
                    Cinv_half_delta = np.linalg.solve(L.T, np.linalg.solve(L, delta))
                except np.linalg.LinAlgError:
                    have_L = False
                    Cinv_half_delta = delta
            else:
                Cinv_half_delta = delta

            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * Cinv_half_delta
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-10, 10.0))

            # Direct covariance update
            drift = 1 - np.exp(-2 * c_1 * np.dot(p_c, p_c))
            C = (1 - c_1 - c_mu) * (L @ L.T)
            C += c_1 * (drift * np.outer(p_c, p_c) + (1 - np.exp(-2 * c_sigma)) * np.eye(dim))
            for i in range(mu_):
                d = selected[i][1] - x_mean
                C += c_mu * w[i] * np.outer(d, d)
            C = 0.5 * (C + C.T)

            # Refresh Cholesky
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                have_L = True
            except np.linalg.LinAlgError:
                L = L + 1e-4 * np.eye(dim)
                have_L = False

            # Stagnation check
            if best_f < prev_best_f - 1e-8:
                stagnation_count = 0
                prev_best_f = best_f
            else:
                stagnation_count += 1

            # If stagnant, restart CMA-ES from best point with fresh covariance
            if stagnation_count >= max_stagnation:
                x_mean = best_x + 0.3 * rng.standard_normal(dim)
                x_mean = np.clip(x_mean, -5.0, 5.0)
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                L = np.eye(dim)
                have_L = True
                stagnation_count = 0
                prev_best_f = best_f

            # If sigma is too small, we've likely converged locally
            if sigma < 1e-6:
                break

        # ---- Phase 3: Nelder-Mead polish from best point ----
        if n_calls < budget and not timed_out():
            remaining = budget - n_calls
            if remaining >= 10:
                nm_calls = min(remaining, 80)
                nelder_mead(best_x, max_calls=nm_calls, init_step=0.1)

        # ---- Phase 4: A few more NM restarts if budget remains ----
        if n_calls < budget and not timed_out():
            temp = 0.5
            for restart in range(5):
                if n_calls >= budget or timed_out():
                    break
                remaining = budget - n_calls
                if remaining < 10:
                    break

                if rng.random() < 0.3:
                    x_start = rng.uniform(-5.0, 5.0, dim)
                else:
                    x_start = best_x + 0.3 * temp * rng.standard_normal(dim)

                x_start = np.clip(x_start, -5.0, 5.0)
                nm_calls = min(remaining, 60)
                nelder_mead(x_start, max_calls=nm_calls, init_step=0.05 * temp + 0.01)
                temp *= 0.6

        return best_x, best_f
# EVOLVE-BLOCK-END
