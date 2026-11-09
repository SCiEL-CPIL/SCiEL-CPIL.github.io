"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        # Phase 1: Coarse random search to find basin of attraction
        # Use stratified random sampling for better coverage
        n_phase1 = 50
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Stratified sampling: divide box into sub-regions
        # For 5D with 50 samples, use a grid-like approach
        n_per_dim = 2  # 2^5 = 32 base grid points, plus random
        grid_points = []
        for idx in range(2 ** dim):
            point = np.array([(idx >> d) & 1 for d in range(dim)], dtype=float)
            # Map from {0,1}^dim to [-5,5]^dim
            point = point * 10.0 - 5.0
            grid_points.append(point)

        # Evaluate grid points
        for p in grid_points:
            if time.time() > deadline:
                break
            p_clipped = np.clip(p, -5, 5)
            f_val = func(p_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p_clipped.copy()

        # Add some random points to fill gaps
        remaining_phase1 = n_phase1 - len(grid_points)
        for _ in range(remaining_phase1):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            x_rand = rng.uniform(-5, 5, dim)
            f_val = func(x_rand)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_rand.copy()

        # Phase 2: CMA-ES refinement from best point found
        if calls_made >= self.budget or time.time() > deadline:
            return best_x, best_f

        # CMA-ES parameters
        x0 = best_x.copy()
        sigma0 = 1.5  # initial step size
        # CMA-ES population size
        lam = 4 + int(3 * np.log(dim))  # standard CMA-ES population size
        mu = lam // 2  # number of parents

        # Initialize CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)  # covariance matrix
        sigma = sigma0
        pc = np.zeros(dim)  # evolution path for covariance
        ps = np.zeros(dim)  # evolution path for step size
        # Precompute constants
        alpha = (2.0 + 2.0 / (dim + 1.0)) / (dim + 4.0 + 2.0 / (dim + 1.0))
        cs = (alpha * (dim + 1.5) + 1.0) / (dim + 5 + alpha * (3.0 + 1.0 / (dim + 1.0)))
        ds = 1.0 + 2.0 * max(0.0, (np.sqrt((dim + 1.0) / (dim + 1.0)) - 1.0) + alpha / 2.0)
        cc = (4.0 + dim / (dim + 4.0)) / ((dim + 2.0) * (dim + 2.0 / (dim + 4.0)))
        cmu = 2.0 / ((dim + 1.3) ** 2 + dim + 2.0)
        cmu = min(cmu, 2.0 - cc)
        # Cholesky factor of C
        cholC = np.linalg.cholesky(C)
        invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
        # Precompute weights
        weights = np.zeros(lam)
        log_w = np.log(np.arange(lam, 0, -1))
        w_sum = np.sum(log_w[:mu])
        weights[:mu] = log_w[:mu] / w_sum
        weights[:mu] -= 1.0 / mu
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # CMA-ES main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                z = rng.standard_normal(dim)
                y = sigma * (cholC @ z)
                x_trial = np.clip(mean + y, -5, 5)
                f_val = func(x_trial)
                calls_made += 1
                pop[i] = x_trial
                pop_f[i] = f_val

                if f_val < best_f:
                    best_f = f_val
                    best_x = x_trial.copy()

            # Sort by fitness
            idx_sorted = np.argsort(pop_f)
            pop_sorted = pop[idx_sorted]
            pop_f_sorted = pop_f[idx_sorted]

            # Update mean
            new_mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
            # Clip mean to box
            new_mean = np.clip(new_mean, -5, 5)

            # Update evolution paths
            y_vec = (new_mean - mean) / sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * y_vec
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** 2) <
                    (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(1.0 - (1.0 - cs) ** 2))
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mu_eff) * y_vec

            # Update covariance
            delta_hsig = (1.0 - hsig) * cc * (2.0 - cc)
            C = (1.0 + cmu * delta_hsig - cc) * C
            for i in range(mu):
                z_i = invsqrtC @ (pop_sorted[i] - mean) / sigma
                C += cmu * weights[i] * np.outer(z_i, z_i)

            # Update sigma
            sigma *= np.exp((1.0 / ds) * (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** 2) - 1.0))
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update Cholesky and inverse square root
            try:
                cholC = np.linalg.cholesky(C)
                invsqrtC = np.linalg.inv(cholC)
            except np.linalg.LinAlgError:
                # Add small regularization if C is not positive definite
                C += 1e-10 * np.eye(dim)
                cholC = np.linalg.cholesky(C)
                invsqrtC = np.linalg.inv(cholC)

            mean = new_mean

            # Check convergence
            if sigma < 1e-10 or np.all(np.abs(pop_f_sorted[:mu] - pop_f_sorted[mu]) < 1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
