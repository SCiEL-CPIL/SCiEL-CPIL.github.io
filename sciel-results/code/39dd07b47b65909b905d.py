"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned) with PCA-initialized covariance."""
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer with PCA-initialized covariance for ill-conditioned landscapes.
    - Initial sampling to estimate search directions
    - PCA-based covariance initialization to align with funnel structure
    - Full CMA-ES with proper covariance adaptation
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin from 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def check():
            return calls_used < budget and time.time() < deadline - 3.0

        # Phase 1: Initial sampling for PCA-based covariance initialization
        # We sample a few points to estimate the landscape orientation.
        n_init = min(30, max(10, budget // 10))
        samples = []
        for i in range(n_init):
            if not check():
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

        # Initialize CMA-ES parameters
        pop_size = max(6, int(4 + 3 * np.log(dim)))
        if pop_size > (budget - calls_used) // 5:
            pop_size = max(6, (budget - calls_used) // 5)

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize mean at best point
        m = best_x.copy()

        # Initialize sigma and Covariance C using PCA on the best samples
        # We want C to reflect the anisotropy of the landscape.
        # We take the top K samples (best fitness) to estimate the local structure.
        K = min(dim, len(samples))
        if K >= 2:
            top_samples = np.array([s[1] for s in samples[:K]])
            # Center the samples
            center = top_samples.mean(axis=0)
            centered = top_samples - center
            
            # Compute covariance
            try:
                cov_mat = np.cov(centered.T)
                if cov_mat.shape == (dim, dim):
                    # Ensure positive definite
                    eigvals, eigvecs = np.linalg.eigh(cov_mat)
                    # Clip small eigenvalues to avoid numerical instability
                    eigvals = np.maximum(eigvals, 1e-6)
                    
                    # We want to initialize C such that it has a similar shape to the data covariance,
                    # but scaled to a reasonable search step size.
                    # The trace of the data covariance gives an idea of the spread.
                    # We normalize the covariance to have unit trace, then scale by an initial sigma^2.
                    # A reasonable initial sigma for [-5,5] is around 1.0-2.0.
                    # Let's use a base sigma of 1.0, so C should have trace ~ dim * 1.0^2? 
                    # Actually, in CMA-ES, y = m + sigma * C^{1/2} z.
                    # If C is unit trace, the effective step size in principal directions varies.
                    # Let's set C to be the normalized covariance matrix.
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    trace_C = np.trace(C)
                    if trace_C > 1e-8:
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

        # Ensure C is positive definite for Cholesky
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            sigma = 1.0
            C_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Phase 2: CMA-ES main loop
        while calls_used + pop_size <= budget and time.time() < deadline - 5:
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
                if calls_used >= budget or time.time() > deadline - 3:
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
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w

            # Update sigma
            if h_sigma:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 0.001, 5.0))

            # Update p_c
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0
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
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            m = np.clip(m, -5, 5)

            # Recompute C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

        return best_x, best_fx
