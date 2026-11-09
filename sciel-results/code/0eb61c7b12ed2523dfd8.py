"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with PCA-initialized covariance for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin from 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def check_time():
            return time.time() < deadline - 5.0

        def check_budget():
            return calls_used < budget

        # Phase 1: Initial sampling for PCA-based covariance initialization
        n_init = min(20, max(10, budget // 20))
        samples = []
        for i in range(n_init):
            if not check_time() or not check_budget():
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            samples.append((fx, x.copy()))

        # Sort samples by fitness
        samples.sort(key=lambda t: t[0])

        # CMA-ES parameters
        pop_size = max(6, int(4 + 3 * np.log(dim)))
        if pop_size > (budget - calls_used) // 8:
            pop_size = max(4, (budget - calls_used) // 8)

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize mean at best point
        m = best_x.copy()

        # Initialize covariance using PCA on best samples
        K = min(dim, len(samples))
        if K >= 2:
            top_samples = np.array([s[1] for s in samples[:K]])
            center = top_samples.mean(axis=0)
            centered = top_samples - center

            try:
                cov_mat = np.cov(centered.T)
                if cov_mat.shape == (dim, dim):
                    eigvals, eigvecs = np.linalg.eigh(cov_mat)
                    eigvals = np.maximum(eigvals, 1e-8)

                    # Normalize to unit trace for CMA-ES
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    trace_C = np.trace(C)
                    if trace_C > 1e-10:
                        C /= trace_C
                    else:
                        C = np.eye(dim) / dim
                    sigma = 1.0
                else:
                    C = np.eye(dim)
                    sigma = 1.0
            except Exception:
                C = np.eye(dim)
                sigma = 1.0
        else:
            C = np.eye(dim)
            sigma = 1.0

        # Ensure positive definite for Cholesky
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            sigma = 1.0
            C_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Phase 2: CMA-ES main loop
        while check_budget() and check_time():
            # Check if we have enough budget for a full population
            if calls_used + pop_size > budget:
                # Do a final single evaluation if possible
                if calls_used < budget:
                    y = m + sigma * (C_sqrt @ rng.standard_normal(dim))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
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
                if not check_budget() or not check_time():
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y.copy()))

            if len(evaluated) < pop_size:
                break

            # Sort by fitness
            evaluated.sort(key=lambda t: t[0])

            # Update best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()

            # Select top mu individuals
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * evaluated[i][1]

            # Calculate step in normalized space
            z_w = np.zeros(dim)
            for i in range(mu):
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                z_w += weights[i] * z_i

            # Update p_sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) <
                           (1.4 + 2 / (dim + 1)) * chi_n) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff) * z_w

            # Update sigma
            if h_sigma:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 0.001, 10.0))

            # Update p_c
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1) ** 2) <
                       (1.4 + 2 / (dim + 1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0
            p_c = (1 - c1) * p_c + h_c * np.sqrt(c1 * (2 - c1) * mu_eff) * z_w

            # Update Covariance C
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
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            m = np.clip(m, -5, 5)

            # Recompute Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sigma = 1.0
                C_sqrt = np.eye(dim)

            # Stagnation detection: restart if no improvement in last 3 generations
            if len(evaluated) > 0 and evaluated[0][0] >= best_fx - 1e-10:
                pass  # Could add restart logic here

        return best_x, best_fx
# EVOLVE-BLOCK-END
