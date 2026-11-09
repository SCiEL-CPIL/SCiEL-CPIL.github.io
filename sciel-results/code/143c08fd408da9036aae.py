"""CMA-ES with restarts for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with PCA-initialized covariance and stagnation-based restarts.
    - PCA-based covariance initialization to align with funnel structure
    - Full CMA-ES with proper covariance adaptation
    - Restart on stagnation to escape local minima
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cmaes(self, func, rng, dim, budget, calls_used, best_fx, best_x,
                   deadline, pop_size=None, C_init=None, sigma_init=None, m_init=None):
        """Run CMA-ES until budget exhausted or stagnation detected."""
        if pop_size is None:
            pop_size = max(6, int(4 + 3 * np.log(dim)))
        if pop_size > max(1, budget - calls_used) // 5:
            pop_size = max(4, (budget - calls_used) // 5)
        if pop_size < 4:
            pop_size = 4

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize parameters
        if m_init is not None:
            m = m_init.copy()
        else:
            m = best_x.copy()

        if C_init is not None:
            C = C_init.copy()
        else:
            C = np.eye(dim)

        if sigma_init is not None:
            sigma = sigma_init
        else:
            sigma = 1.0

        # Ensure C is positive definite
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            sigma = 1.0

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        stagnation_counter = 0
        stagnation_threshold = 3 * pop_size  # 3 generations without improvement
        best_fx_local = best_fx
        best_x_local = best_x.copy()

        # Main CMA-ES loop
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
            improved = False
            if evaluated[0][0] < best_fx_local:
                best_fx_local = evaluated[0][0]
                best_x_local = evaluated[0][1].copy()
                improved = True
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Check for stagnation
            if stagnation_counter >= stagnation_threshold:
                break

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

        return calls_used, best_fx_local, best_x_local, improved

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
        n_init = min(40, budget // 10)
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

        # Compute PCA-based covariance from initial samples
        C_init = None
        sigma_init = None
        if len(samples) >= dim:
            top_samples = np.array([s[1] for s in samples[:min(15, len(samples))]])
            centered = top_samples - top_samples.mean(axis=0)
            try:
                cov_init = np.cov(centered.T)
                if cov_init.shape == (dim, dim):
                    eigvals, eigvecs = np.linalg.eigh(cov_init)
                    eigvals = np.maximum(eigvals, 1e-8)
                    C_init = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_init /= np.trace(C_init)
                    sigma_init = float(np.clip(np.sqrt(np.trace(C_init) / dim), 0.1, 2.0))
            except Exception:
                C_init = None
                sigma_init = None

        # Phase 2: CMA-ES with restarts
        restart_count = 0
        max_restarts = 5

        while calls_used < budget and time.time() < deadline - 10 and restart_count < max_restarts:
            # Run CMA-ES
            calls_used, best_fx, best_x, improved = self._run_cmaes(
                func, rng, dim, budget, calls_used, best_fx, best_x,
                deadline, C_init=C_init, sigma_init=sigma_init, m_init=best_x
            )

            if not improved or calls_used >= budget or time.time() > deadline - 10:
                break

            restart_count += 1

            # For restart, use the best point as new center with reduced sigma
            # and reset covariance to identity (or keep PCA if still useful)
            C_init = None  # Reset to identity for restart
            sigma_init = 0.5  # Smaller step size for refinement

        return best_x, best_fx
