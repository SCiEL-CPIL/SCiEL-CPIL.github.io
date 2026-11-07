"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES-like evolution strategy with adaptive step size for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with stratified sampling
        # Use a deterministic set of well-spread points to find the basin
        n_global = min(50, budget // 10)
        rng = np.random.default_rng(42)

        # Stratified sampling in 5D
        # Use a grid with random jitter for better coverage
        per_dim = max(2, int(round(n_global ** (1.0 / dim))))
        grid = np.linspace(0, 1, per_dim + 1)[:-1] + 0.5 / per_dim
        # Add jitter
        grid = grid + rng.uniform(-0.5 / per_dim, 0.5 / per_dim, per_dim)
        grid = np.clip(grid, 0, 1)

        # Generate grid points
        mesh = np.meshgrid(*([grid] * dim), indexing='ij')
        grid_points = np.array([m.ravel() for m in mesh]).T
        # Shuffle and take n_global points
        idx = rng.permutation(len(grid_points))[:n_global]
        global_points = grid_points[idx] * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        call_count = 0

        # Evaluate global points
        for x in global_points:
            if time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: evaluate center
            x = np.zeros(dim)
            fx = func(x)
            call_count += 1
            best_x = x
            best_f = fx

        # Phase 2: CMA-ES-like local optimization
        # Use an evolution strategy with adaptive step size
        if time.time() > deadline:
            return best_x, best_f

        # Initialize CMA-ES parameters
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Step size
        sigma = 1.0
        # Mean
        mean = best_x.copy()
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Eigen decomposition cache
        B = np.eye(dim)
        D = np.ones(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Parameters
        cmu = 1.0 / (4 * dim + lambda_)
        c_sigma = 1.0 / (dim + 4)
        c_c = (4 * dim + lambda_ - 2) / ((dim + 2) * (lambda_ + 4 - 2))
        mu = lambda_ // 2
        # Weights
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Evaluate initial population
        population = []
        fitness = []
        for i in range(lambda_):
            if time.time() > deadline:
                break
            # Generate sample
            z = rng.standard_normal(dim)
            x = mean + sigma * (B @ (D * z))
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            population.append(x)
            fitness.append(fx)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if len(population) == 0:
            return best_x, best_f

        # Main CMA-ES loop
        while call_count < budget and time.time() < deadline:
            # Sort by fitness
            order = np.argsort(fitness)
            population = np.array(population)[order]
            fitness = np.array(fitness)[order]

            # Update mean
            new_mean = np.sum(w[:, np.newaxis] * population[:mu], axis=0)

            # Update covariance matrix
            y = (new_mean - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ y)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count - lambda_)))) < (1 + 2.0 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)

            C = (1 + cmu * delta_hsig * (c_c * (2 - c_c) - 1) + cmu) * C \
                - cmu * (np.outer(y, y) + (c_c * (2 - c_c) - 1) * C)

            # Ensure positive definiteness
            try:
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))
                B = B[:, D > 1e-10]
                D = D[D > 1e-10]
            except:
                B = np.eye(dim)
                D = np.ones(dim)

            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(B.T @ (D * (B.T @ (y)))) - chi_n))
            sigma = np.clip(sigma, 1e-8, 10.0)

            mean = new_mean

            # Generate new population
            new_population = []
            new_fitness = []
            for i in range(lambda_):
                if time.time() > deadline or call_count >= budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                fx = func(x)
                call_count += 1
                new_population.append(x)
                new_fitness.append(fx)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(new_population) == 0:
                break

            population = new_population
            fitness = new_fitness

            # Check convergence
            if len(fitness) > 1 and np.ptp(fitness[:mu]) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
