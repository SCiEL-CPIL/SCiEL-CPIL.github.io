"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based search for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0=1.5, max_iter=100):
            nonlocal calls_made, best_f, best_x

            n = dim
            lambda_ = 10  # population size

            # CMA-ES parameters
            mu = lambda_ // 2  # number of recombination weights
            # Weights: log-linear weights
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step-size adaptation rate
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            # Covariance matrix adaptation rate
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            # Damping for covariance matrix
            d_c = c_c * (mu_eff + 2) / (n + mu_eff + 5)

            # Initialize
            mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)

            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            def cholesky_decompose(A):
                """Compute Cholesky decomposition with fallback."""
                try:
                    L = np.linalg.cholesky(A)
                    return L
                except np.linalg.LinAlgError:
                    # Add small regularization
                    eps = 1e-12
                    A_reg = A + eps * np.eye(n)
                    try:
                        L = np.linalg.cholesky(A_reg)
                        return L
                    except np.linalg.LinAlgError:
                        # Last resort: eigenvalue decomposition
                        eigvals, eigvecs = np.linalg.eigh(A)
                        eigvals = np.maximum(eigvals, eps)
                        L = eigvecs @ np.diag(np.sqrt(eigvals))
                        return L

            def sample_from_cov(mean, sigma, C):
                """Sample from multivariate normal with given mean, sigma, covariance."""
                L = cholesky_decompose(C)
                z = np.random.randn(n)
                return mean + sigma * (L @ z)

            # Initial population
            pop = []
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    return
                x = sample_from_cov(mean, sigma, C)
                x = np.clip(x, -5.0, 5.0)
                f = eval_func(x)
                pop.append((x, f))

            for iteration in range(max_iter):
                if calls_made >= budget - lambda_ or time.time() > deadline:
                    return

                # Sort by function value
                pop.sort(key=lambda t: t[1])
                selected = pop[:mu]

                # Weighted mean of selected points
                new_mean = np.zeros(n)
                for i, (x, f) in enumerate(selected):
                    new_mean += weights[i] * x

                # Update evolution path for step-size
                C_sqrt_inv = np.linalg.inv(cholesky_decompose(C))
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                          (C_sqrt_inv @ (new_mean - mean)) / sigma

                # Update evolution path for covariance
                p_c = (1 - c_c) * p_c + \
                      np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                      (new_mean - mean) / sigma

                # Heuristic h_sigma
                h_sigma = np.linalg.norm(p_sigma) / \
                          np.sqrt(1 - (1 - c_sigma) ** (iteration + 1)) / \
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))
                h_sigma = 1.0 if h_sigma < 1.4 + 2 / (n + 1) else 0.0

                # Update covariance matrix
                delta_h = (h_sigma - 1) * c_c * (2 - c_c)
                outer = np.outer(p_c, p_c)
                C = (1 + c_c * (2 - c_c) + c_sigma * (2 - c_sigma) * mu_eff -
                     c_sigma * mu_eff) * C + \
                    c_sigma * mu_eff * (outer + delta_h * np.outer(p_sigma, p_sigma)) + \
                    c_sigma * (2 - c_sigma) * mu_eff * \
                    (np.outer(p_sigma, p_sigma) - C)

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 0):
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = (eigvecs * eigvals) @ eigvecs.T

                # Update mean
                mean = new_mean

                # Step-size adaptation
                sigma *= np.exp(c_sigma / c_sigma *
                                (np.linalg.norm(p_sigma) /
                                 np.sqrt(1 - (1 - c_sigma) ** (iteration + 1)) -
                                 np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))))

                # Clip sigma
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Generate new population
                pop = []
                for i in range(lambda_):
                    if calls_made >= budget or time.time() > deadline:
                        return
                    x = sample_from_cov(mean, sigma, C)
                    x = np.clip(x, -5.0, 5.0)
                    f = eval_func(x)
                    pop.append((x, f))

        # Multi-start CMA-ES with diverse initializations
        rng = np.random.default_rng(42)
        starts = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
            np.array([2.5, -2.5, 2.5, -2.5, 2.5])[:dim],
            np.array([-2.5, 2.5, -2.5, 2.5, -2.5])[:dim],
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for start in starts:
            if calls_made >= budget - 15 or time.time() > deadline:
                break
            cma_es(start, sigma0=1.5, max_iter=40)

        # Final refinement around the best point
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline:
            cma_es(best_x.copy(), sigma0=0.5, max_iter=30)

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
