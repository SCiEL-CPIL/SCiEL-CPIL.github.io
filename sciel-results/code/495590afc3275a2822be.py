"""Hybrid Bayesian Optimization + CMA-ES with adaptive restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer combining Bayesian Optimization (GP + EI) for global exploration
    with CMA-ES adaptive restarts for exploitation and (1+1)-ES for local polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0
        rng = np.random.default_rng(42)

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        # --- GP Surrogate ---
        def gp_predict(X, X_train, y_train, length_scale, signal_variance, noise_variance):
            n_train = X_train.shape[0]
            n_pred = X.shape[0]

            K_train = np.zeros((n_train, n_train))
            for i in range(n_train):
                for j in range(n_train):
                    diff = X_train[i] - X_train[j]
                    dist_sq = np.dot(diff, diff)
                    K_train[i, j] = signal_variance * np.exp(-0.5 * dist_sq / (length_scale**2))
            K_train += noise_variance * np.eye(n_train)

            K_pred_train = np.zeros((n_pred, n_train))
            for i in range(n_pred):
                for j in range(n_train):
                    diff = X[i] - X_train[j]
                    dist_sq = np.dot(diff, diff)
                    K_pred_train[i, j] = signal_variance * np.exp(-0.5 * dist_sq / (length_scale**2))

            try:
                L = np.linalg.cholesky(K_train + 1e-8 * np.eye(n_train))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_train + 1e-8 * np.eye(n_train), y_train, rcond=None)[0]

            mean = K_pred_train @ alpha
            v = np.linalg.solve(L, K_pred_train.T)
            var = signal_variance - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-10)

            return mean, var

        def expected_improvement(mean, std, best_f_val, xi=0.01):
            std = np.maximum(std, 1e-10)
            improvement = best_f_val - mean - xi
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            return np.maximum(ei, 0)

        # --- CMA-ES Implementation ---
        def cma_es(x0, sigma0, pop_size, max_calls, sigma_min=1e-10, sigma_max=10.0):
            nonlocal n_calls, best_x, best_f

            x = np.asarray(x0, dtype=float).copy()
            sigma = sigma0
            calls_used = 0
            N = dim

            mu = max(2, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (N + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
            c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
            c_1 = 2 / ((N + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
            chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

            pc = np.zeros(N)
            ps = np.zeros(N)
            C = np.eye(N)

            best_local_x = x.copy()
            best_local_f = np.inf
            stagnant_generations = 0
            last_best_f = np.inf

            for gen in range(max_calls // pop_size + 1):
                if n_calls >= budget or check_time():
                    break

                population = []
                fitnesses = []
                for i in range(pop_size):
                    if n_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(N)
                    y = x + sigma * z
                    fy, y_eval = call_func(y)
                    if fy is None:
                        continue
                    population.append(y_eval)
                    fitnesses.append(fy)
                    calls_used += 1

                if len(population) < 2:
                    break

                order = np.argsort(fitnesses)
                population = [population[i] for i in order]
                fitnesses = [fitnesses[i] for i in order]

                gen_best_f = fitnesses[0]
                if gen_best_f < best_local_f:
                    best_local_f = gen_best_f
                    best_local_x = population[0].copy()
                    if gen_best_f < best_f:
                        best_f = gen_best_f
                        best_x = population[0].copy()
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1

                if stagnant_generations > 15:
                    break

                x_new = np.zeros(N)
                for i in range(min(mu, len(population))):
                    x_new += weights[i] * population[i]
                x = x_new

                y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(N)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_N * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1)))) < (1.4 + 2 / (N + 1)) * chi_N
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_mean

                delta_d = (1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1)) * c_sigma
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((N, N))
                for i in range(min(mu, len(population))):
                    zi = (population[i] - x_new) / sigma if sigma > 0 else np.zeros(N)
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c_1 * delta_d - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * (c_c * (2 - c_c) * np.eye(N) - C)) + c_mu * rank_mu
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(N)

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
                sigma = max(sigma_min, min(sigma_max, sigma))

                if sigma < sigma_min or np.all(np.abs(C) < 1e-10):
                    break

            return best_local_x, best_local_f, calls_used

        # --- (1+1)-ES Polisher ---
        def local_polish(x0, f0, max_calls, sigma_init=0.3):
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            f = f0
            sigma = sigma_init
            calls_used = 0
            stagnant_count = 0

            for _ in range(max_calls):
                if n_calls + 1 > budget or check_time():
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy, y_eval = call_func(y)
                if fy is None:
                    break
                calls_used += 1

                if fy < f:
                    x = y_eval
                    f = fy
                    sigma *= 1.05
                    stagnant_count = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    stagnant_count += 1
                    if stagnant_count > 25:
                        break

                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 3.0)

            return x, f, calls_used

        # --- Phase 1: Initial exploration with random sampling + BO ---
        n_init = min(8, budget)
        X_train = np.zeros((n_init, dim))
        y_train = np.zeros(n_init)

        for i in range(n_init):
            if n_calls >= budget or check_time():
                break
            x_new = rng.uniform(-5, 5, dim)
            f_new, x_eval = call_func(x_new)
            if f_new is None:
                break
            X_train[i] = x_eval
            y_train[i] = f_new
            update_best(x_eval, f_new)

        if best_x is None:
            return None

        # --- Phase 2: BO loop with CMA-ES restarts on stagnation ---
        length_scale = 1.0
        signal_variance = max(np.var(y_train), 0.1)
        noise_variance = 1e-6

        bo_stagnant = 0
        bo_max_stagnant = 10
        cma_restart_count = 0
        max_cma_restarts = 3

        while n_calls < budget and not check_time():
            # Generate candidates via LHS
            n_candidates = 300
            X_candidates = rng.uniform(-5, 5, (n_candidates, dim))

            try:
                mean_pred, var_pred = gp_predict(X_candidates, X_train, y_train, length_scale, signal_variance, noise_variance)
            except Exception:
                x_new = rng.uniform(-5, 5, dim)
                f_new, x_eval = call_func(x_new)
                if f_new is None:
                    break
                update_best(x_eval, f_new)
                X_train = np.vstack([X_train, x_eval])
                y_train = np.append(y_train, f_new)
                continue

            ei = expected_improvement(mean_pred, np.sqrt(var_pred), best_f, xi=0.01)
            idx = np.argmax(ei)
            x_new = X_candidates[idx]

            f_new, x_eval = call_func(x_new)
            if f_new is None:
                break

            improved = update_best(x_eval, f_new)
            X_train = np.vstack([X_train, x_eval])
            y_train = np.append(y_train, f_new)

            if improved:
                bo_stagnant = 0
                length_scale *= 1.03
            else:
                bo_stagnant += 1
                length_scale *= 0.97
                length_scale = max(length_scale, 0.2)

            if len(y_train) > 1:
                signal_variance = max(np.var(y_train), 0.1)

            # If BO stagnates, trigger CMA-ES restart from best point
            if bo_stagnant >= bo_max_stagnant and cma_restart_count < max_cma_restarts:
                bo_stagnant = 0
                cma_restart_count += 1

                # Use CMA-ES from current best with moderate sigma
                remaining = budget - n_calls
                if remaining > 30:
                    pop_size = min(10, max(4, remaining // 5))
                    cma_calls = min(remaining - 10, 100)
                    _, _, _ = cma_es(best_x, 1.0, pop_size, cma_calls)

                    # Reset BO training set to recent points
                    n_keep = min(len(X_train), 20)
                    if n_keep > 0:
                        X_train = X_train[-n_keep:]
                        y_train = y_train[-n_keep:]
                        signal_variance = max(np.var(y_train), 0.1)
                        length_scale = 0.8

        # --- Phase 3: Final (1+1)-ES polish ---
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining > 5:
                local_polish(best_x, best_f, remaining, sigma_init=0.2)

        return best_x, best_f

# EVOLVE-BLOCK-END
