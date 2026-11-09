"""CMA-ES with adaptive restarts and GP-based local refinement for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts and Gaussian Process refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def cma_es(x0, sigma0, max_calls, rng):
            """
            Runs CMA-ES with per-evaluation stagnation tracking.
            Returns (best_x, best_f, calls_used, stagnated).
            """
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            local_best_x = x.copy()
            local_best_f = np.inf
            calls_used = 0
            stagnation_counter = 0
            no_improve_threshold = 5 * lam

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    return local_best_x, local_best_f, calls_used, True
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    return local_best_x, local_best_f, calls_used, True
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Main loop
            while calls_used < max_calls and not check_time():
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                x_new = np.sum(weights[:, None] * pop[:mu], axis=0)

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        return local_best_x, local_best_f, calls_used, True
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                if stagnation_counter > no_improve_threshold:
                    return local_best_x, local_best_f, calls_used, True

                if sigma < 1e-10:
                    return local_best_x, local_best_f, calls_used, True

            return local_best_x, local_best_f, calls_used, True

        def gp_refine(x_center, f_center, n_points, rng):
            """
            Simple GP-based local refinement using Expected Improvement.
            Uses a simple radial basis kernel.
            Returns (best_x, best_f, calls_used).
            """
            nonlocal best_x, best_f, n_calls
            n = dim
            X = [x_center.copy()]
            Y = [f_center]
            local_best_x = x_center.copy()
            local_best_f = f_center
            calls_used = 0
            epsilon = 1e-8
            length_scale = 1.0
            signal_var = 1.0

            def kernel(X1, X2):
                """Matern-5/2 kernel."""
                d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
                sqrt_5d2 = np.sqrt(5.0 * d2) / length_scale
                return signal_var * (1.0 + sqrt_5d2 + d2 / 3.0) * np.exp(-sqrt_5d2)

            for _ in range(n_points):
                if n_calls >= budget or check_time():
                    break

                # Build GP model
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                K = kernel(X_arr, X_arr) + epsilon * np.eye(len(X_arr))

                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_arr))
                except np.linalg.LinAlgError:
                    K_reg = K + 1e-6 * np.eye(len(X_arr))
                    try:
                        L = np.linalg.cholesky(K_reg)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_arr))
                    except np.linalg.LinAlgError:
                        break

                # Generate candidate points
                n_candidates = 50
                candidates = local_best_x + rng.uniform(-2.0, 2.0, (n_candidates, n))
                candidates = np.clip(candidates, -5, 5)

                # Compute GP mean and variance at candidates
                K_star = kernel(candidates, X_arr)
                mean = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = signal_var - np.sum(v ** 2, axis=1)
                var = np.maximum(var, 1e-10)
                std = np.sqrt(var)

                # Expected Improvement
                f_best = min(Y_arr)
                z = (f_best - mean) / std
                from scipy.stats import norm
                EI = std * (z * norm.cdf(z) + norm.pdf(z))

                # Select best candidate
                idx = np.argmax(EI)
                x_new = candidates[idx]
                f_new, x_eval = call_func(x_new)
                if f_new is None:
                    break
                calls_used += 1
                X.append(x_eval.copy())
                Y.append(f_new)

                if f_new < local_best_f:
                    local_best_f = f_new
                    local_best_x = x_eval.copy()
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_eval.copy()

            return local_best_x, local_best_f, calls_used

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        initial_candidates = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim),
        ]

        for x in initial_candidates:
            f, x_eval = call_func(x)
            if f is not None:
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()

        if n_calls >= budget or check_time():
            return best_x, best_f

        # Phase 2: CMA-ES with adaptive restarts
        current_x = best_x.copy()
        current_sigma = 1.5
        restart_count = 0
        max_restarts = 5

        while n_calls < budget and not check_time() and restart_count < max_restarts:
            remaining = budget - n_calls
            if remaining < 10:
                break

            if restart_count == max_restarts - 1:
                alloc = remaining
            else:
                alloc = int(remaining * 0.6)

            alloc = max(alloc, 20)

            res_x, res_f, used, stagnated = cma_es(current_x, current_sigma, alloc, rng)

            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

            restart_count += 1

            if not stagnated:
                current_x = res_x.copy()
                current_sigma = max(0.5, current_sigma * 0.7)
            else:
                # Perturb for next restart
                current_x = best_x + rng.uniform(-1.0, 1.0, dim)
                current_sigma = 1.5

        # Phase 3: GP-based local refinement with remaining budget
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining >= 10:
                gp_n = min(remaining, 30)
                _, _, _ = gp_refine(best_x, best_f, gp_n, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
