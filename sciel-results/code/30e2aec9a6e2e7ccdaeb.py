import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    for BBOB f13 (Sharp Ridge).
    
    CMA-ES is a state-of-the-art derivative-free optimizer that adapts its
    search distribution to the local landscape geometry. For sharp ridge
    functions, it naturally learns the anisotropic structure (narrow in one
    direction, wide in others) through covariance matrix adaptation,
    making it far more efficient than coordinate descent or Nelder-Mead
    for this problem class.
    
    Strategy:
    1. Brief global exploration with random sampling to identify the basin.
    2. Run CMA-ES with adaptive population size and sigma management.
    3. Use the best point from CMA-ES as the final answer.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(12345)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val is None or np.isnan(f_val) or np.isinf(f_val):
                    return None
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration - find a good starting basin
        n_random = 15
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0.copy()
            best_f = f0

        # Phase 2: CMA-ES
        def cma_es(x_mean, sigma_init, max_iter=100):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            # CMA-ES parameters
            n = dim
            # Population size: default is 4 + floor(3 * ln(n))
            lam = max(4 + int(3 * np.log(n)), 2 * n)
            # Cap lambda to not waste too many calls
            lam = min(lam, 20)

            # Initial parameters
            mu = lam // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Step size control
            cs = 1.0 / (n + 3) * (1 + 2 * max(0, (mueff - 1) / (n + 3)))
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 3)) + cs

            # Covariance matrix adaptation
            c1 = (mueff + 2) / ((n + 2) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # State variables
            x_mean = x_mean.copy()
            sigma = sigma_init
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)  # Cholesky of C^{-1/2}
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Cholesky decomposition of C (for sampling)
            def cholesky(A):
                try:
                    return np.linalg.cholesky(A)
                except np.linalg.LinAlgError:
                    # Regularize
                    A_reg = A + 1e-10 * np.eye(n)
                    return np.linalg.cholesky(A_reg)

            L = cholesky(C)

            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return

                # Generate population
                y = rng.standard_normal((lam, n))
                x_pop = np.array([x_mean + sigma * (L @ y[i]) for i in range(lam)])
                x_pop = np.clip(x_pop, -5.0, 5.0)

                # Evaluate
                f_pop = []
                for i in range(lam):
                    f_val = safe_call(x_pop[i])
                    if f_val is None:
                        return
                    f_pop.append(f_val)
                f_pop = np.array(f_pop)

                # Sort by fitness
                order = np.argsort(f_pop)
                x_pop_sorted = x_pop[order]
                f_pop_sorted = f_pop[order]

                # Update best
                if f_pop_sorted[0] < best_f:
                    best_f = f_pop_sorted[0]
                    best_x = x_pop_sorted[0].copy()

                # Check convergence
                if np.std(f_pop_sorted[:mu]) < 1e-12:
                    break

                # Update mean
                x_new_mean = np.sum(weights[:, None] * x_pop_sorted[:mu], axis=0)

                # Update sigma (CSA)
                y_mean = np.sum(weights[:, None] * (x_pop_sorted[:mu] - x_mean), axis=0)
                y_mean = y_mean / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (iteration + 1))) / chi_n < 1.4 + 2 / (n + 1)
                hsig = float(hsig)

                # Update C (PCA)
                p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mueff) * (x_new_mean - x_mean) / sigma

                # Rank-one update
                C += c1 * (2 / (1 - c1) - 1 + cmu) * C - c1 * np.outer(p_c, p_c)

                # Rank-mu update
                if cmu > 0:
                    y_inv = np.array([(x_pop_sorted[i] - x_mean) / sigma for i in range(mu)])
                    rank_mu = cmu * np.sum(weights[i] * np.outer(y_inv[i], y_inv[i]) for i in range(mu))
                    C += rank_mu

                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update Cholesky
                L = cholesky(C)

                # Update sigma
                sigma *= np.exp((cs / ds) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-15, 50.0)

                # Update mean
                x_mean = x_new_mean

                # Check if we should stop
                if sigma < 1e-10 or np.all(np.abs(x_mean) > 4.9):
                    break

        # Run CMA-ES from the best found point
        # Start with a moderate sigma
        sigma_start = 1.0
        cma_es(best_x, sigma_start, max_iter=80)

        # If we have budget left, try a second CMA-ES run from a slightly perturbed best
        if calls < budget - 10 and time.time() < deadline:
            x_restart = best_x + rng.standard_normal(dim) * 0.1
            x_restart = np.clip(x_restart, -5.0, 5.0)
            f_restart = safe_call(x_restart)
            if f_restart is not None:
                cma_es(x_restart, 0.5, max_iter=40)

        return best_x, best_f
