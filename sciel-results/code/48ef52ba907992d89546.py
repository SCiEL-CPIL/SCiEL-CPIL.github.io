"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    - Random sampling phase to find a promising region.
    - CMA-ES with covariance matrix adaptation for anisotropic landscapes.
    - Final polish phase for tight local refinement.
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

        # Phase 1: Random sampling to find a good initial point and estimate covariance
        n_samples = min(50, budget // 10)
        samples = []
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            samples.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # Estimate covariance from top samples
        samples_sorted = sorted(samples, key=lambda t: t[0])
        n_top = max(5, len(samples) // 4)
        top_samples = np.array([s[1] for s in samples_sorted[:n_top]])
        
        # Compute covariance of top samples
        if n_top > 1:
            cov = np.cov(top_samples, rowvar=False)
            # Regularize to ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            cov = np.eye(dim)

        # Phase 2: CMA-ES
        best_x = best_x.copy()
        best_fx = best_fx

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda
        sigma = 1.0
        C = cov  # covariance matrix
        # Precompute Cholesky-like decomposition for sampling
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)
        
        # Mean update parameters
        c_mean = 1.0 / (pop_size + 3)
        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Track state
        m = best_x.copy()
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        success = 0
        stall_count = 0
        max_stall = 30

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append((y, i))

            # Evaluate offspring
            evaluated = []
            for y, idx in offspring:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y, idx))

            if len(evaluated) < pop_size:
                break

            # Sort by fitness
            evaluated.sort(key=lambda t: t[0])

            # Update best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Select top mu individuals
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * evaluated[i][1]
            
            # Step size adaptation
            z = np.zeros(dim)
            for i in range(mu):
                z += weights[i] * (C_sqrt.T @ (evaluated[i][1] - m) / sigma)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            sigma = float(np.clip(sigma, 0.001, 10.0))

            # Update covariance
            for i in range(mu):
                y_i = (evaluated[i][1] - m) / sigma
                delta_s = y_w - m / sigma
                C = (1 + c1 * (cmu - 1) - c1 - 2 * cmu) * C
                C += c1 * (np.outer(delta_s, delta_s) + np.eye(dim))
                for j in range(mu):
                    C += cmu * weights[j] * np.outer(y_i, y_i)
                    y_i = (evaluated[j][1] - m) / sigma

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            # Clip mean to box
            m = np.clip(m, -5, 5)

            # Recompute C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            # Restart if stalled
            if stall_count >= max_stall:
                # Reset covariance to identity and increase sigma
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma = max(sigma * 2.0, 1.0)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                max_stall = min(max_stall + 15, 80)

        # Phase 3: Final polish with small sigma CMA-ES
        if calls_used < budget and time.time() < deadline - 2.0:
            sigma = 0.05
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            m = best_x.copy()
            p_sigma = np.zeros(dim)
            stall_count = 0
            max_stall = 50

            pop_size = max(4, int(4 + 3 * np.log(dim)))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)
            c_sigma = (pop_size + 2) / (dim + 4)
            c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
            d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            while calls_used < budget:
                if time.time() > deadline - 1.0:
                    break

                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = m + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append((y, i))

                evaluated = []
                for y, idx in offspring:
                    if calls_used >= budget or time.time() > deadline - 1.0:
                        break
                    fy = func(y)
                    calls_used += 1
                    evaluated.append((fy, y, idx))

                if len(evaluated) < pop_size:
                    break

                evaluated.sort(key=lambda t: t[0])

                if evaluated[0][0] < best_fx:
                    best_fx = evaluated[0][0]
                    best_x = evaluated[0][1].copy()
                    stall_count = 0
                else:
                    stall_count += 1

                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * evaluated[i][1]

                z = np.zeros(dim)
                for i in range(mu):
                    z += weights[i] * (C_sqrt.T @ (evaluated[i][1] - m) / sigma)

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

                sigma = float(np.clip(sigma, 0.001, 0.5))

                for i in range(mu):
                    y_i = (evaluated[i][1] - m) / sigma
                    delta_s = y_w - m / sigma
                    C = (1 + c1 * (cmu - 1) - c1 - 2 * cmu) * C
                    C += c1 * (np.outer(delta_s, delta_s) + np.eye(dim))
                    for j in range(mu):
                        C += cmu * weights[j] * np.outer(y_i, y_i)
                        y_i = (evaluated[j][1] - m) / sigma

                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                m = y_w
                m = np.clip(m, -5, 5)

                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)

                if stall_count >= max_stall:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    sigma = max(sigma * 2.0, 0.1)
                    p_sigma = np.zeros(dim)
                    stall_count = 0
                    max_stall = min(max_stall + 20, 100)

        return best_x, best_fx
# EVOLVE-BLOCK-END
