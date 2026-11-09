"""Combined optimizer: multi-start (1+1)-ES exploration with CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start adaptive (1+1)-ES with CMA-ES refinement for BBOB f21.
    
    Phase 1: Diverse multi-start random sampling to identify promising basins.
    Phase 2: Adaptive (1+1)-ES from the best point found in Phase 1.
    Phase 3: CMA-ES refinement from the global best if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        n = dim

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_deadline():
            return eval_count >= budget or time.time() > deadline

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with diverse random sampling
        # Use ~30% of budget for exploration
        n_explore = min(30, max(10, budget // 3))
        
        for i in range(n_explore):
            if check_deadline():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if best_x is None:
            # Should not happen, but safety
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        # Phase 2: Adaptive (1+1)-ES refinement
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        stagnation = 0
        max_stagnation = 40
        improvement_threshold = 1e-10

        while not check_deadline():
            y = x + sigma * rng.standard_normal(n)
            fy = evaluate(y)

            if fy < fx - improvement_threshold:
                # Improvement found
                x = y
                fx = fy
                sigma *= 1.15
                stagnation = 0
            else:
                # No improvement
                sigma *= 0.8
                stagnation += 1

                # Reset sigma if stuck for too long
                if stagnation >= max_stagnation:
                    sigma = max(0.3, min(3.0, sigma * 1.5))
                    stagnation = 0

        # Phase 3: CMA-ES refinement from the best point if budget remains
        remaining = budget - eval_count
        if remaining > 20 and time.time() < deadline - 5:
            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(n))  # 8 for n=5
            mu_ = lambda_ // 2
            weights = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            c_sigma = (mu_eff + 2) / (n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 - mu_eff + c_sigma) / (n + 4.3)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))

            x = best_x.copy()
            sigma = 0.5  # Smaller initial step for refinement
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            while not check_deadline():
                # Generate lambda_ candidates
                ys = []
                fs = []
                
                for _ in range(lambda_):
                    if check_deadline():
                        break
                    
                    try:
                        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    except np.linalg.LinAlgError:
                        L = np.linalg.cholesky(C + 1e-6 * np.eye(n))

                    z = rng.standard_normal(n)
                    y = x + sigma * (L @ z)
                    fy = evaluate(y)
                    ys.append(y)
                    fs.append(fy)

                if len(ys) < mu_:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in order])
                fs_sorted = np.array([fs[i] for i in order])

                # Update mean
                x_old = x.copy()
                x = np.sum(weights[:mu_][:, np.newaxis] * ys_sorted[:mu_], axis=0)

                # Compute L^{-1}(x - x_old) / sigma for path updates
                diff = x - x_old
                try:
                    v = np.linalg.solve(L, diff / sigma)
                except np.linalg.LinAlgError:
                    v = np.linalg.lstsq(L, diff / sigma, rcond=None)[0]

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * v
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(n) - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 5.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * v

                # Rank-mu update
                y_diffs = ys_sorted[:mu_] - x_old
                try:
                    y_diffs_L_inv = np.linalg.solve(L, y_diffs.T).T
                except np.linalg.LinAlgError:
                    y_diffs_L_inv = np.linalg.lstsq(L, y_diffs.T, rcond=None)[0].T

                C = (1 - c_1 - c_mu) * C + \
                    c_1 * np.outer(p_c, p_c) + \
                    c_mu * np.sum(weights[i] * np.outer(y_diffs_L_inv[i], y_diffs_L_inv[i]) for i in range(mu_))

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
