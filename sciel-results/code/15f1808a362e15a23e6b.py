"""Combined program: Diagonal CMA-ES (Rank-1 + Rank-mu) with Nelder-Mead polish for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal CMA-ES variant with rank-1 + rank-mu updates and NM polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 285.0  # Safety margin vs 330s kill

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

        def nelder_mead(x0, max_calls, init_step=0.1):
            """
            Nelder-Mead implementation with call budget tracking.
            Returns early if budget or time is exhausted.
            """
            nonlocal n_calls, best_x, best_f

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += init_step

            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return
                fvals[i] = f

            # Sort simplex by fvals
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

            sort_simplex()

            calls_used = dim + 1
            calls_limit = max_calls

            for _ in range(max_calls):
                if n_calls >= budget or timed_out():
                    break

                # Check convergence
                if np.std(fvals) < 1e-8:
                    break

                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                calls_used += 1

                # Sort to see where x_r fits
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
                    calls_used += 1
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
                    calls_used += 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        sigma_shrink = 0.5
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return
                            fvals[i] = f
                            calls_used += 1
                        sort_simplex()

                if calls_used >= calls_limit:
                    break

        def diagonal_cma_es(x_start, sigma_init, max_calls):
            """
            Diagonal CMA-ES with Rank-1 + Rank-mu updates.
            Uses per-dimension step sizes instead of full covariance.
            """
            nonlocal n_calls, best_x, best_f

            # Population size
            lambda_ = 14
            mu_ = 7
            w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Strategy constants
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
            c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
            c_1 = 2 / (dim ** 2 + 6)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State
            x_mean = x_start.copy()
            sigmas = np.full(dim, sigma_init)  # Per-dimension step sizes
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            calls_used = 0
            best_in_run = float('inf')
            stagnation_counter = 0
            max_stagnation = 30

            while calls_used < max_calls and n_calls < budget and not timed_out():
                # Generate population
                offspring = []
                for _ in range(lambda_):
                    if n_calls >= budget or timed_out():
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigmas * z
                    f = call_f(y)
                    if f is None:
                        break
                    offspring.append((f, y, z))
                    calls_used += 1

                if len(offspring) < mu_:
                    break

                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu_]

                # Check stagnation
                if selected[0][0] < best_in_run - 1e-10:
                    best_in_run = selected[0][0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter > max_stagnation:
                    break

                # Update mean
                old_mean = x_mean.copy()
                x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
                delta = x_mean - old_mean

                # Update evolution paths
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * delta
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                    np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

                # Update per-dimension sigma
                sigma_factor = np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigmas = sigmas * sigma_factor
                sigmas = np.clip(sigmas, 1e-10, 20.0)

                # Rank-1 + Rank-mu update for diagonal covariance
                drift = 1 - np.exp(-2 * c_1 * np.dot(p_c, p_c))
                diag_C = sigmas ** 2
                diag_C = (1 - c_1 - c_mu) * diag_C
                diag_C += c_1 * (drift * p_c ** 2 + (1 - np.exp(-2 * c_sigma)) * sigmas ** 2)
                for i in range(mu_):
                    d = selected[i][1] - x_mean
                    diag_C += c_mu * w[i] * d ** 2

                # Update sigmas from diagonal
                sigmas = np.sqrt(np.maximum(diag_C, 1e-20))
                sigmas = np.clip(sigmas, 1e-10, 20.0)

                # Stagnation guard: if all sigmas are tiny, re-seed
                if np.all(sigmas < 1e-7):
                    x_mean = best_x + 0.3 * rng.standard_normal(dim)
                    sigmas = np.full(dim, 0.5)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    break

            return x_mean, best_in_run

        # Phase 1: Initial diverse exploration to break Discus ring symmetry
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

        # Phase 2: Diagonal CMA-ES main search
        remaining = budget - n_calls
        if remaining > 50:
            # Use 70% of remaining budget for diagonal CMA-ES
            cma_budget = int(remaining * 0.7)
            diagonal_cma_es(best_x.copy(), sigma_init=1.5, max_calls=cma_budget)

        # Phase 3: Nelder-Mead polish from best point found
        remaining = budget - n_calls
        if remaining > 10:
            nm_budget = min(remaining, 50)
            nelder_mead(best_x.copy(), max_calls=nm_budget, init_step=0.1)

        # Phase 4: If budget remains, do a few more NM restarts
        temp = 0.5
        for _ in range(3):
            if n_calls >= budget or timed_out():
                break
            x_start = best_x + temp * rng.standard_normal(dim)
            x_start = np.clip(x_start, -5.0, 5.0)
            remaining = budget - n_calls
            if remaining < 10:
                break
            nelder_mead(x_start, max_calls=min(remaining, 20), init_step=0.05)
            temp *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
