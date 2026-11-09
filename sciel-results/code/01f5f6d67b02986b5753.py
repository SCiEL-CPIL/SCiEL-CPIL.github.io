"""Hybrid GP-guided exploration + CMA-ES local refinement for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Hybrid optimizer: GP for global exploration, CMA-ES for local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or check_time():
                return None, None
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            n_calls += 1
            try:
                f = float(func(x_clipped))
            except Exception:
                return None, None
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f, x_clipped

        # ---- GP helper ----
        def gp_predict(X_train, Y_train, X_test, length_scale, sigma_f, noise=1e-6):
            X_train = np.atleast_2d(X_train)
            X_test = np.atleast_2d(X_test)
            Y_mean = np.mean(Y_train)
            Y_std = np.std(Y_train)
            if Y_std < 1e-10:
                Y_std = 1.0
            Y_norm = (Y_train - Y_mean) / Y_std

            def kernel(X1, X2, ls, sf):
                d2 = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
                d2 = np.maximum(d2, 0)
                return (sf**2) * np.exp(-d2 / (2 * ls**2))

            K = kernel(X_train, X_train, length_scale, sigma_f)
            K[np.diag_indices_from(K)] += noise
            K_s = kernel(X_train, X_test, length_scale, sigma_f)
            try:
                alpha = np.linalg.solve(K, Y_norm)
                mu = K_s.T @ alpha
                v = np.linalg.solve(K, K_s)
                var = sigma_f**2 - np.sum(K_s**2 * v, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                mu = np.full(X_test.shape[0], Y_mean)
                var = np.full(X_test.shape[0], Y_std**2)
            mu = mu * Y_std + Y_mean
            var = var * Y_std**2
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, best_y, xi=0.0):
            s = np.sqrt(var)
            z = (best_y - mu - xi) / (s + 1e-10)
            ei = (best_y - mu - xi) * norm.cdf(z) + s * norm.pdf(z)
            return np.maximum(ei, 0)

        # ---- CMA-ES helper ----
        def cma_es(x0, sigma0, max_calls, local_rng):
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu_sel = lam // 2

            weights = np.log(mu_sel + 0.5) - np.log(np.arange(1, mu_sel + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0)**2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            local_best_x = x.copy()
            local_best_f = np.inf
            calls_used = 0
            stagnation = 0
            no_improve_threshold = 5 * lam

            # Initial population
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    return local_best_x, local_best_f, calls_used, True
                z = local_rng.standard_normal(n)
                f, x_eval = call_func(x + sigma * z)
                if f is None:
                    return local_best_x, local_best_f, calls_used, True
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
                    stagnation = 0
                else:
                    stagnation += 1

            while calls_used < max_calls and not check_time():
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                x_new = np.sum(weights[:, None] * pop[:mu_sel], axis=0)

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                hsig = (np.linalg.norm(ps) / np.sqrt(max(1e-10, 1.0 - (1.0 - cs)**(2 * max(1, calls_used // lam))))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu_sel):
                    y_i = (pop[i] - x) / max(sigma, 1e-15)
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = local_rng.standard_normal(n)
                    f, x_eval = call_func(x + sigma * (Cinv_sqrt @ z))
                    if f is None:
                        return local_best_x, local_best_f, calls_used, True
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        stagnation = 0
                    else:
                        stagnation += 1

                if stagnation > no_improve_threshold or sigma < 1e-10:
                    return local_best_x, local_best_f, calls_used, True

            return local_best_x, local_best_f, calls_used, True

        # ---- Phase 1: Initial exploration with diverse points ----
        n_initial = min(25, budget)
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                initial_points.append(pt)
        # Random
        n_rand = n_initial - len(initial_points)
        if n_rand > 0:
            initial_points.extend(rng.uniform(-5, 5, (n_rand, dim)).tolist())

        X_data = []
        Y_data = []
        for pt in initial_points:
            if check_time() or n_calls >= budget:
                break
            f, x_eval = call_func(pt)
            if f is None:
                break
            X_data.append(x_eval)
            Y_data.append(f)

        # ---- Phase 2: Hybrid GP exploration + CMA-ES exploitation ----
        # Track history for stagnation
        recent_best = best_f
        stagnation_window = 40
        history_best = []

        phase = "explore"  # explore or exploit
        cma_budget_remaining = 0
        cma_active = False

        while n_calls < budget and not check_time():
            # Record current best for stagnation tracking
            history_best.append(best_f)
            if len(history_best) > stagnation_window:
                history_best.pop(0)

            # Stagnation check
            if len(history_best) >= stagnation_window:
                window_best = min(history_best)
                if best_f - window_best < 1e-6:
                    # Stagnated - trigger CMA-ES if not already, or restart exploration
                    if not cma_active and best_x is not None:
                        phase = "exploit"
                        cma_active = True
                        # Estimate initial sigma from recent spread
                        if len(X_data) >= 2:
                            spread = np.std(X_data[-min(20, len(X_data)):], axis=0)
                            avg_spread = np.mean(spread[spread > 1e-4]) if np.any(spread > 1e-4) else 1.0
                        else:
                            avg_spread = 1.0
                        cma_sigma = max(0.2, min(avg_spread, 2.0))
                        cma_budget_remaining = min(80, budget - n_calls)
                    else:
                        # If CMA-ES already used or failed, do GP-guided restart
                        phase = "explore"
                        cma_active = False

            if phase == "exploit" and cma_active:
                # Run CMA-ES from current best
                cma_rng = np.random.default_rng(int(time.time() * 1000) % 100000)
                cma_calls = min(cma_budget_remaining, budget - n_calls)
                if cma_calls >= 10:
                    cma_best_x, cma_best_f, cma_used, cma_stagnated = cma_es(
                        best_x, cma_sigma, cma_calls, cma_rng
                    )
                    # Update global best (already done in call_func)
                    cma_active = False
                    phase = "explore"
                    # If CMA-ES made progress, continue exploring from new best
                    if cma_best_f < recent_best - 1e-8:
                        recent_best = cma_best_f
                else:
                    cma_active = False
                    phase = "explore"

            # GP-guided exploration
            if check_time() or n_calls >= budget:
                break

            n_propose = min(3, budget - n_calls)
            n_candidates = 150

            candidates = []
            # Random
            candidates.append(rng.uniform(-5, 5, (n_candidates // 2, dim)))
            # Near best
            if best_x is not None:
                candidates.append(np.clip(best_x + rng.standard_normal((n_candidates // 3, dim)) * 0.8, -5, 5))
            # GP-informed random (spread)
            candidates.append(rng.uniform(-5, 5, (n_candidates - n_candidates // 2 - n_candidates // 3, dim)))

            all_cands = np.vstack(candidates)

            if len(X_data) >= 3:
                X_arr = np.array(X_data)
                Y_arr = np.array(Y_data)
                data_spread = np.std(X_arr, axis=0)
                avg_spread = np.mean(data_spread[data_spread > 1e-6]) if np.any(data_spread > 1e-6) else 1.0
                length_scale = max(0.5, min(avg_spread, 3.0))
                sigma_f = max(np.std(Y_arr), 0.1)

                mu, var = gp_predict(X_arr, Y_arr, all_cands, length_scale, sigma_f)
                ei = expected_improvement(mu, var, best_f)
                top_idx = np.argsort(ei)[::-1][:n_propose]
                selected = all_cands[top_idx]
            else:
                selected = all_cands[:n_propose]

            for pt in selected:
                if n_calls >= budget or check_time():
                    break
                f, x_eval = call_func(pt)
                if f is None:
                    break
                X_data.append(x_eval)
                Y_data.append(f)

        return best_x, best_f
# EVOLVE-BLOCK-END
