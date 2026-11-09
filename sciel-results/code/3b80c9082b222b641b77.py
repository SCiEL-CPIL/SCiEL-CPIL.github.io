"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Uses covariance matrix adaptation to handle anisotropic landscapes.
    - Initial random sampling to find a promising region.
    - CMA-ES with proper update rules for the ill-conditioned funnel.
    - Time guard to stay well under the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find a good initial point
        n_samples = min(20, budget // 10)
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        x = best_x.copy()
        f_best = best_fx
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Termination criteria
        max_iterations = budget - calls_used
        stall_count = 0
        max_stall = 50

        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break

            # Generate population
            ys = []
            for i in range(lambda_):
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                ys.append(y)

            # Evaluate population
            fs = []
            for y in ys:
                fy = func(y)
                calls_used += 1
                fs.append(fy)

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update best
            if fs_sorted[0] < f_best:
                f_best = fs_sorted[0]
                x = ys_sorted[0].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Update mean
            x_new = x + sum(weights[i] * (ys_sorted[i] - x) for i in range(mu))
            x = x_new

            # Compute evolution paths
            y_mean = np.mean([ys_sorted[i] for i in range(mu)], axis=0, weights=weights[:mu]) if mu > 0 else np.zeros(dim)
            y_mean = sum(weights[i] * (ys_sorted[i] - x) for i in range(mu))
            y_mean /= sigma

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            p_sigma_norm = np.linalg.norm(p_sigma)

            h_sigma = 1 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1) * C
            C -= c_1 * (np.outer(p_c, p_c) + delta_h * C)
            for i in range(mu):
                z_i = (ys_sorted[i] - x) / sigma
                C += c_mu * weights[i] * (np.outer(z_i, z_i) - C)

            # Ensure C is symmetric positive definite
            C = 0.5 * (C + C.T)
            try:
                C_sqrt, _ = np.linalg.sqrtm(C)
                if not np.all(np.isfinite(C_sqrt)):
                    raise ValueError
            except:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-10, 100.0))

            # Restart if stalled
            if stall_count >= max_stall:
                # Reset CMA-ES with larger sigma
                sigma = max(2.0, sigma * 2.0)
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                max_stall = min(max_stall + 20, 100)

                # Try a random restart
                x_new = x + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_used += 1
                if f_new < f_best:
                    x = x_new.copy()
                    f_best = f_new

        # Phase 3: Final polish with small (1+1)-ES
        if calls_used < budget and time.time() < deadline - 2.0:
            sigma = 0.05
            successes = 0
            total_trials = 0
            window_size = 10
            while calls_used < budget:
                if time.time() > deadline - 1.0:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                total_trials += 1

                if fy < f_best:
                    x = y.copy()
                    f_best = fy
                    successes += 1

                if total_trials >= window_size:
                    rate = successes / total_trials
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma /= 1.2
                    sigma = float(np.clip(sigma, 0.001, 0.5))
                    successes = 0
                    total_trials = 0

        return x, f_best
# EVOLVE-BLOCK-END
