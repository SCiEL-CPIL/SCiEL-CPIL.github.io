"""Combined optimizer: LHS multi-start + GP-guided BO + CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """LHS multi-start with GP-guided Bayesian Optimization and CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0
        X_obs = []
        f_obs = []

        def check_time():
            return time.time() > deadline - 5

        def record(x, f):
            nonlocal best_x, best_f, call_count
            X_obs.append(x.copy())
            f_obs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling for diverse initial exploration
        n_lhs = 25
        # Generate LHS points in [0,1]^n then map to [-4,4]^n
        lhs_points = []
        for i in range(n_lhs):
            pt = np.zeros(n)
            for d in range(n):
                pt[d] = (i + rng.uniform()) / n_lhs
            lhs_points.append(pt)
        lhs_points = np.array(lhs_points) * 8.0 - 4.0  # map to [-4,4]
        
        # Add center and some structured points
        extra_points = [np.zeros(n)]
        for d in range(n):
            for val in [-3.0, 3.0]:
                pt = np.zeros(n)
                pt[d] = val
                extra_points.append(pt.copy())
        
        all_starts = list(lhs_points) + extra_points
        # Shuffle and take first n_lhs + 2
        rng.shuffle(np.array(all_starts))
        all_starts = [np.array(p) for p in all_starts[:n_lhs + 2]]

        for sp in all_starts:
            if call_count >= budget - 1 or check_time():
                break
            x = np.clip(sp.copy(), -5, 5)
            fx = func(x)
            call_count += 1
            record(x, fx)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            call_count += 1
            record(best_x, best_f)

        # Phase 2: GP-guided Bayesian Optimization
        # Use ~50% of remaining budget for BO
        remaining = budget - call_count
        bo_budget = int(remaining * 0.5)

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-8

        def gp_predict(X_new):
            """Predict mean and variance at X_new using simple GP."""
            X_train = np.array(X_obs)
            y_train = np.array(f_obs)
            m = len(y_train)
            
            # Normalize y
            y_mean = np.mean(y_train)
            y_std = max(np.std(y_train), 1e-8)
            y_norm = (y_train - y_mean) / y_std
            
            # RBF kernel
            def rbf_kernel(X1, X2):
                X1 = np.atleast_2d(X1)
                X2 = np.atleast_2d(X2)
                sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return signal_var * np.exp(-0.5 * sq_dists / (length_scale ** 2))
            
            K = rbf_kernel(X_train, X_train) + noise_var * np.eye(m)
            K_star = rbf_kernel(X_new, X_train)
            
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(m))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                v = np.linalg.solve(L, K_star.T)
                mu = y_mean + v.T @ alpha * y_std
                var = signal_var - v.T @ v
                var = np.maximum(var, 1e-10)
                return mu, np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback: return best known value with high uncertainty
                return np.full(len(np.atleast_2d(X_new)), best_f), np.full(len(np.atleast_2d(X_new)), 1.0)

        def acquisition_function(X_cand):
            """Expected Improvement acquisition function."""
            X_cand = np.atleast_2d(X_cand)
            mu, sigma = gp_predict(X_cand)
            f_best = min(f_obs)
            z = (mu - f_best) / np.maximum(sigma, 1e-8)
            ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # minimize negative EI

        # BO loop
        bo_iterations = 0
        stagnant_count = 0
        prev_best = best_f

        for _ in range(bo_budget):
            if call_count >= budget - 1 or check_time():
                break

            # Generate candidate points: mix of random and around current best
            n_cand = 50
            candidates = []
            
            # Random points in [-4, 4]
            rand_cands = rng.uniform(-4, 4, (n_cand // 2, n))
            candidates.append(rand_cands)
            
            # Points around current best with decreasing radius
            radius = max(0.5, 2.0 * (1.0 - bo_iterations / max(bo_budget, 1)))
            local_cands = best_x + radius * rng.standard_normal((n_cand // 2, n))
            local_cands = np.clip(local_cands, -4, 4)
            candidates.append(local_cands)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, -5, 5)
            
            # Evaluate acquisition function
            acq_vals = acquisition_function(candidates)
            best_cand_idx = np.argmin(acq_vals)
            x_new = candidates[best_cand_idx]
            
            fx_new = func(x_new)
            call_count += 1
            record(x_new, fx_new)
            bo_iterations += 1
            
            # Track stagnation
            if fx_new < prev_best - 1e-10:
                prev_best = fx_new
                stagnant_count = 0
            else:
                stagnant_count += 1
            
            # Adaptively adjust length scale if stuck
            if stagnant_count > 10:
                length_scale = max(0.5, length_scale * 0.7)
                stagnant_count = 0

        # Phase 3: CMA-ES refinement from best point
        if call_count < budget - 5 and not check_time():
            lambda_ = 4 + int(3 * np.log(n))  # 8 for n=5
            mu_ = lambda_ // 2  # 4
            weights = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (n + 4.3)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))

            x = best_x.copy()
            sigma = 0.3
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            L = np.linalg.cholesky(C + 1e-10 * np.eye(n))

            while call_count < budget:
                if check_time():
                    break

                ys = []
                fs = []
                for _ in range(lambda_):
                    if call_count >= budget or check_time():
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    call_count += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < mu_:
                    break

                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])

                x_old = x.copy()
                x = np.sum(weights[:mu_][:, np.newaxis] * ys_sorted[:mu_], axis=0)

                diff = x - x_old
                try:
                    v = np.linalg.solve(L, diff / sigma)
                except np.linalg.LinAlgError:
                    v = np.linalg.lstsq(L, diff / sigma, rcond=None)[0]

                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * v
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(n) - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 5.0)

                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * v

                y_diffs = ys_sorted[:mu_] - x_old
                try:
                    y_diffs_L_inv = np.linalg.solve(L, y_diffs.T).T
                except np.linalg.LinAlgError:
                    y_diffs_L_inv = np.linalg.lstsq(L, y_diffs.T, rcond=None)[0].T

                C = (1 - c_1 - c_mu) * C + \
                    c_1 * np.outer(p_c, p_c) + \
                    c_mu * np.sum(weights[i] * np.outer(y_diffs_L_inv[i], y_diffs_L_inv[i]) for i in range(mu_))

                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    L = np.linalg.cholesky(C + 1e-6 * np.eye(n))

        return best_x, best_f
# EVOLVE-BLOCK-END
