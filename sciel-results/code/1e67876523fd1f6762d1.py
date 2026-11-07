"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and time guard for step ellipsoidal function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        # Count function calls
        call_count = [0]

        def counted_func(x):
            call_count[0] += 1
            return func(x)

        rng = np.random.default_rng(0)

        # Initial point - try center first, then random
        x0 = np.zeros(dim)
        f0 = counted_func(x0)
        best_x = x0.copy()
        best_f = f0

        # Try a few random starts to find a good initial basin
        for _ in range(5):
            x_try = rng.uniform(-5, 5, dim)
            f_try = counted_func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # standard CMA-ES population size
        # Step size initialization
        sigma = 1.0
        # Mean
        m = best_x.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Learning rates
        cmu = (mu + 2) / ((dim + 1.3) ** 2 + mu)
        cc = (4 + dim / lambda_) / ((dim + 4) ** 2 + lambda_)
        cs = (mu + 2 + 2 * mu / lambda_) / ((dim + 2) + mu + 2 * mu / lambda_)
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cs
        # Weights
        w = np.log(np.arange(1, lambda_ + 1)) - np.log(np.arange(1, lambda_ + 1)).mean()
        w = w / w.sum()
        mu = int(lambda_ / 2)
        mu_eff = 1 / np.sum(w[:mu] ** 2)
        cc = (4 + dim / lambda_) / ((dim + 4) ** 2 + lambda_)
        cs = (mu_eff + 2 + 2 * mu_eff / lambda_) / ((dim + 2) + mu_eff + 2 * mu_eff / lambda_)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cmu = (2 - cmu + 2 * mu_eff / lambda_) / ((dim + 2) ** 2 + mu_eff)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Cholesky of C
        C_sqrt = np.linalg.cholesky(C)

        def cholesky_update(C, C_sqrt, w, y, mu_eff, dim):
            """Update Cholesky factorization."""
            C_new = C
            for i in range(mu_eff):
                C_new += 2 * cmu * w[i] * np.outer(y[i], y[i])
            C_new *= (1 - cmu)
            # Ensure positive definite
            C_new = (C_new + C_new.T) / 2
            try:
                C_sqrt_new = np.linalg.cholesky(C_new)
                return C_new, C_sqrt_new
            except np.linalg.LinAlgError:
                # Add small regularization
                C_new += 1e-10 * np.eye(dim)
                C_sqrt_new = np.linalg.cholesky(C_new)
                return C_new, C_sqrt_new

        # Main CMA-ES loop
        while call_count[0] < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            ys = []
            xs = []
            fs = []
            for i in range(lambda_):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x_new = m + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = counted_func(x_new)
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)

            if len(xs) == 0:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Update mean
            m_new = m + sigma * sum(w[i] * ys[i] for i in range(mu_eff))
            m_new = np.clip(m_new, -5, 5)

            # Update evolution paths
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv @ (m_new - m) / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (call_count[0] // lambda_))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1) else 0
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (m_new - m) / sigma

            # Update covariance
            ys_mu = np.array(ys[:mu_eff])
            C, C_sqrt = cholesky_update(C, C_sqrt, w, ys_mu, mu_eff, dim)

            # Update step size
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            m = m_new

        # Final local polish: coordinate descent
        for _ in range(3):
            if call_count[0] >= budget or time.time() > deadline:
                break
            for d in range(dim):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                # Try moving in each direction
                for delta in [-0.5, -0.1, 0.1, 0.5]:
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    x_try = best_x.copy()
                    x_try[d] += delta
                    x_try = np.clip(x_try, -5, 5)
                    f_try = counted_func(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
