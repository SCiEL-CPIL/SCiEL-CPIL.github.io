"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Robust CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    - Random sampling phase for initialization.
    - Corrected CMA-ES implementation with covariance matrix adaptation.
    - Stagnation detection with restarts (reset covariance, increase sigma).
    - Time guard to stay within wall-clock limits.
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
        n_samples = min(50, budget // 10)
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # Phase 2: CMA-ES with restarts
        # CMA-ES Parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        m = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        stall_count = 0
        max_stall = 20
        restart_sigma_mult = 1.5

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y))

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
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * evaluated[i][1]

            # Calculate step in normalized space
            # z_w = sum w_i * (y_i - m) / sigma * C^{-1/2}
            # We have C_sqrt, so C^{-1/2} = C_sqrt^{-1} = C_sqrt.T^{-1}
            # Actually, y_i = m + sigma * C_sqrt @ z_i => z_i = C_sqrt^{-1} (y_i - m) / sigma
            # We need z_w = sum w_i * z_i
            z_w = np.zeros(dim)
            for i in range(mu):
                # Solve C_sqrt @ z_i = (y_i - m) / sigma
                # Using np.linalg.solve for stability
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                z_w += weights[i] * z_i

            # Update p_sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
            
            # Update sigma
            if h_sigma:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            sigma = float(np.clip(sigma, 0.001, 10.0))

            # Update p_c
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0
            p_c = (1 - c1) * p_c + h_c * np.sqrt(c1 * (2 - c1) * mu_eff) * z_w

            # Update Covariance C
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1 - h_c) * c1 * (1 - c1) * C) + cmu * sum w_i y_i y_i^T
            # Note: y_i here is in normalized space (z_i)
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_c) * c1 * (1 - c1) * C)
            
            for i in range(mu):
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                C += cmu * weights[i] * np.outer(z_i, z_i)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            m = np.clip(m, -5, 5)

            # Recompute C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            # Restart logic
            if stall_count >= max_stall:
                # Reset covariance to identity, increase sigma
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                sigma = max(sigma * restart_sigma_mult, 1.0)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                max_stall = min(max_stall + 10, 60)
                restart_sigma_mult *= 1.2

        # Phase 3: Final polish with small sigma CMA-ES or (1+1)-ES
        # If we have time and budget, do a quick local search
        if calls_used < budget and time.time() < deadline - 2.0:
            sigma = 0.05
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            m = best_x.copy()
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            stall_count = 0
            max_stall = 20

            while calls_used < budget:
                if time.time() > deadline - 1.0:
                    break

                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = m + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                evaluated = []
                for y in offspring:
                    if calls_used >= budget or time.time() > deadline - 1.0:
                        break
                    fy = func(y)
                    calls_used += 1
                    evaluated.append((fy, y))

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

                z_w = np.zeros(dim)
                for i in range(mu):
                    try:
                        z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                    except np.linalg.LinAlgError:
                        z_i = (evaluated[i][1] - m) / sigma
                    z_w += weights[i] * z_i

                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w
                if h_sigma:
                    sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = float(np.clip(sigma, 0.001, 0.5))

                h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0
                p_c = (1 - c1) * p_c + h_c * np.sqrt(c1 * (2 - c1) * mu_eff) * z_w

                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(p_c, p_c) + (1 - h_c) * c1 * (1 - c1) * C)
                for i in range(mu):
                    try:
                        z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                    except np.linalg.LinAlgError:
                        z_i = (evaluated[i][1] - m) / sigma
                    C += cmu * weights[i] * np.outer(z_i, z_i)

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
                    p_c = np.zeros(dim)
                    stall_count = 0
                    max_stall = min(max_stall + 10, 50)

        return best_x, best_fx
# EVOLVE-BLOCK-END
