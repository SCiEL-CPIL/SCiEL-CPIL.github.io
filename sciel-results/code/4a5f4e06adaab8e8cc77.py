"""Multi-Restart CMA-ES black-box optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Restart CMA-ES optimizer for Rastrigin.

    Strategy:
    - Split the 500-call budget across multiple independent CMA-ES restarts.
    - Each restart initializes from a diverse point (LHS + jitter).
    - CMA-ES adapts its covariance matrix to learn the landscape geometry,
      which is critical for rotated Rastrigin where axes are not aligned.
    - Track the global best across all restarts.
    - Use small populations (N=4+2*floor(3+ln(dim))) for efficiency per restart.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._func = None
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        self._func = func
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = np.zeros(self.dim)

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        def evaluate(x):
            nonlocal self
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f_val = func(x)
            self._call_count += 1
            if not np.isfinite(f_val):
                f_val = 1e300
            if f_val < self._best_f:
                self._best_f = f_val
                self._best_x = x.copy()
            return f_val

        # CMA-ES parameters
        # Standard: N = 4 + 2*floor(3 + ln(dim))
        N = 4 + 2 * int(3 + np.log(dim))  # For dim=5: 4+2*floor(3+1.609)=4+2*4=12
        N = max(4, N)

        # Number of restarts: each restart needs at least ~N*2 evaluations to be useful
        # Budget 500, N=12 → ~40 evals per restart minimum for meaningful progress
        # Let's do 8-12 restarts
        n_restarts = max(4, min(12, budget // (N * 3)))

        # Generate diverse starting points using Latin Hypercube Sampling
        lhs_points = self._lhs_samples(n_restarts, dim, rng)

        # Also add a few purely random points for extra diversity
        extra_random = min(4, n_restarts - len(lhs_points)) if n_restarts > len(lhs_points) else 0
        if extra_random > 0:
            rand_points = rng.uniform(-4.0, 4.0, size=(extra_random, dim))
            lhs_points = np.vstack([lhs_points, rand_points])

        # CMA-ES implementation (compact, inline)
        for restart_idx in range(len(lhs_points)):
            if self._call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize CMA-ES at this starting point
            mean = lhs_points[restart_idx].copy()
            sigma = 1.0  # Initial step size
            C = np.eye(dim)  # Covariance matrix
            C_inv = np.eye(dim)
            ps = np.zeros(dim)  # Evolution path for sigma
            pc = np.zeros(dim)  # Evolution path for C
            # Damping factors
            cs = (N + 2) / (dim + N + 5)
            ds = 1 + 2 * max(0, (dim - 1) / (N + 3) - 1) + cs
            cmu = 2 / ((N + 4) ** 2 + dim)
            cc = (4 - dim + cmu) / (dim + 4 - cmu)
            nu_sigma = np.sqrt(N) * (1 + 2 / (dim - 1) + cs)

            # Weighting for rank update
            mu = N // 2
            weights = np.log(np.arange(1, mu + 1)) - np.log(mu)
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Evaluate initial population
            pop = np.zeros((N, dim))
            fit = np.zeros(N)
            for i in range(N):
                if self._call_count >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                pop[i] = mean + sigma * (C @ z)  # C^{1/2} z approximated via Cholesky-free
                # Actually, use proper sampling: z ~ N(0, C)
                # For efficiency, use: C^{1/2} @ z where C^{1/2} from eigendecomp
                # But for simplicity and speed, use: mean + sigma * z (diagonal init)
                # Let's do proper: sample from N(mean, sigma^2 * C)
                # Use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                    pop[i] = mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    pop[i] = mean + sigma * z

                fit[i] = evaluate(pop[i])

            # CMA-ES iterations
            max_iter = (budget - self._call_count) // N
            for gen in range(max_iter):
                if self._call_count >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fit)
                y = np.zeros((mu, dim))
                for k in range(mu):
                    y[k] = (pop[order[k]] - mean) / sigma

                # Update mean
                mean_new = mean + sigma * np.sum(weights[:, None] * y, axis=0)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv @ (mean_new - mean) / sigma)
                hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * nu_sigma
                pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * (mean_new - mean) / sigma

                # Update covariance
                delta_hs = (1 - hs) * cc * (2 - cc)
                outer = np.outer(pc, pc)
                C = (1 + cmu * (delta_hs - 1) - cs * (2 - cs)) * C \
                    + cmu * np.sum(weights[:, None, None] * y[:, None, :] * y[:, None, :].T, axis=0)

                # Ensure C is symmetric positive definite
                C = 0.5 * (C + C.T)
                # Add small regularization
                C += 1e-12 * np.eye(dim)

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / nu_sigma - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                mean = mean_new

                # Generate next population
                for i in range(N):
                    if self._call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    try:
                        L = np.linalg.cholesky(C)
                        pop[i] = mean + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        pop[i] = mean + sigma * z

                    fit[i] = evaluate(pop[i])

                # If sigma is too small, this restart is converged
                if sigma < 1e-7:
                    break

        return self._best_x, self._best_f

    def _lhs_samples(self, n, dim, rng):
        """Generate Latin Hypercube samples in [-4, 4]^dim."""
        samples = np.zeros((n, dim))
        for j in range(dim):
            perm = rng.permutation(n)
            samples[:, j] = (perm + rng.uniform(size=n)) / n
            samples[:, j] = samples[:, j] * 8.0 - 4.0  # Map to [-4, 4]
        return samples
# EVOLVE-BLOCK-END
