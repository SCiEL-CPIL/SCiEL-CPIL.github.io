<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.87
        return best_x, best_fx
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-Based CMA-ES for bent cigar optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # State variables
        x = np.zeros(dim)  # mean
        sigma = 3.0  # initial step size (cover the box)
        C = np.eye(dim)  # covariance matrix
        B = np.eye(dim)  # eigenvectors
        D = np.ones(dim)  # eigenvalues
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best solution
        best_x = x.copy()
        best_fx = np.inf

        # Evaluate initial mean
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Precompute B^T C B for efficiency
        def update_eigendecomp(C):
            nonlocal B, D
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            return B, D

        B, D = update_eigendecomp(C)

        while calls_made < budget:
            if time.time() >= deadline:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Transform z using B and D
                y = B @ (D * z)
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                population.append(xi)

            # Evaluate population
            fvals = []
            for xi in population:
                fi = func(xi)
                fvals.append(fi)
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()
                calls_made += 1

            # Sort by fitness
            idx = np.argsort(fvals)
            sorted_pop = [population[i] for i in idx]
            sorted_fvals = [fvals[i] for i in idx]

            # Compute recombination mean
            x_old = x.copy()
            x_new = sum(weights[i] * sorted_pop[i] for i in range(mu))

            # Update path variables
            C_sqrt_inv = (B / D) @ B.T  # C^(-1/2)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_new - x_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ (x_new - x_old) / sigma

            # Update sigma (CSA)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance matrix
            v = x_new - x_old
            delta_sigma = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma * d_sigma
            C = (1 + c_c * (2 - c_c) + c_c * delta_sigma) * C \
                - c_c * C @ np.outer(p_c, p_c) @ C \
                + (c_c * (2 - c_c) + c_c * delta_sigma) * np.outer(p_c, p_c) \
                + (1 + c_c * (2 - c_c) + c_c * delta_sigma - 1) * np.outer(v, v) / sigma ** 2

            # Actually, let me use the standard CMA-ES covariance update
            # h_sigma = 1 if ||p_sigma|| / chi_n < (1 + c_sigma) * (1 + 2 / (dim + 1)) else 0
            h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < (1 + c_sigma) * (1 + 2 / (dim + 1)) else 0

            # Standard CMA-ES covariance update
            C = (1 - c_c * (2 - c_c) - c_c * delta_sigma) * C \
                + (c_c * (2 - c_c) + c_c * delta_sigma) * np.outer(p_c, p_c) \
                + (1 - c_c * (2 - c_c) - c_c * delta_sigma) * (
                    np.outer(v, v) / sigma ** 2 - C)

            # Hmm, let me use a cleaner implementation
            # Reset and use proper CMA-ES update
            C = (1 + c_c * (2 - c_c) + c_c * delta_sigma) * C \
                - c_c * (C @ np.outer(p_c, p_c) @ C) \
                + (1 + c_c * (2 - c_c) + c_c * delta_sigma) * np.outer(p_c, p_c) \
                + (1 - 1) * 0  # placeholder

            # Let me just use a simpler but correct covariance update
            # The standard CMA-ES update is:
            # C <- (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1-h_sigma)*c_c(2-c_c)C) + cmu * sum w_i (y_i - x_old)(y_i - x_old)^T
            c1 = (2 + 2 * mu_eff / (dim + 2)) / ((dim + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[0] - 1) / ((dim + 2) ** 2 + mu_eff))

            # Recompute properly
            C = (1 - c1 - cmu) * C \
                + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                + cmu * sum(weights[i] * np.outer(sorted_pop[i] - x_old, sorted_pop[i] - x_old) / sigma ** 2
                            for i in range(mu))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update eigen-decomposition
            B, D = update_eigendecomp(C)

            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
