"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for BBOB f17 with robust parameter settings."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Generate initial points using Sobol sequence
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=dim, scramble=True, seed=123)
        n_init = 32
        sobol_points = sobol.random(n_init)
        sobol_points = sobol_points * 10.0 - 5.0

        # Add center and a few random points
        center = np.zeros(dim)
        rng = np.random.default_rng(42)
        extra = rng.uniform(-5, 5, size=(16, dim))

        initial_points = np.vstack([sobol_points, center.reshape(1, -1), extra])

        # Evaluate all initial points to find the best starting point
        best_x = None
        best_f = np.inf
        for i in range(len(initial_points)):
            if time.time() > deadline:
                break
            x = initial_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Use CMA-ES for the main optimization
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma0 = 2.0  # initial step size

        # Implement CMA-ES manually for better control
        x = best_x.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (pop_size + 2) / (dim + 4)
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
        cmu = 2 / ((dim + 1.3) ** 2 + pop_size)
        cc = (4 - dim + pop_size) / (dim + 4 - pop_size)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x_cma = best_x.copy()
        best_f_cma = best_f
        calls_made = len(initial_points)
        remaining_budget = budget - calls_made

        # CMA-ES loop
        generation = 0
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            W = np.exp(np.log(pop_size / 2) * np.ones(pop_size))
            W /= np.sum(W)
            mu = pop_size // 2
            nu = 0.5 * np.sum(W[:mu])

            # Sample new points
            z = rng.standard_normal((pop_size, dim))
            # Apply C transformation: B = C^(1/2)
            # For simplicity, use Cholesky or eigendecomposition
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim)

            y = (z @ L.T)
            x_population = x + sigma * y

            # Evaluate population
            f_values = np.zeros(pop_size)
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                f_values[i] = func(x_population[i])
                calls_made += 1
                if f_values[i] < best_f_cma:
                    best_f_cma = f_values[i]
                    best_x_cma = x_population[i].copy()

            if calls_made >= budget or time.time() > deadline:
                break

            # Sort by function value
            idx = np.argsort(f_values)
            x_selected = x_population[idx[:mu]]
            f_selected = f_values[idx[:mu]]

            # Update center of mass
            x_new = np.sum(W[:mu, None] * x_selected, axis=0)

            # Update step-size control variable
            xbar_old = x
            x = x_new
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu * nu) * (L.T @ (x - xbar_old) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (generation + 1)) / chiN < (1.4 + 2 / (dim + 1))

            # Update covariance matrix
            if hsig:
                pc = (1 - cs) * pc + cc * (x - xbar_old) / sigma
            C = (1 + cmu * nu * hsig - cmu - cmu * nu * (1 - ds)) * C + cmu * np.sum(
                W[:mu, None, None] * np.outer(
                    x_selected - xbar_old, x_selected - xbar_old
                ), axis=0
            ) + cmu * nu * (1 - ds) * np.outer(pc, pc)

            # Update step size
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))

            # Ensure C is positive semi-definite
            C = 0.5 * (C + C.T)

            generation += 1

            # Check for convergence
            if sigma < 1e-12 or np.all(np.abs(x - xbar_old) < 1e-10):
                break

        # Final result
        if best_f_cma < best_f:
            best_f = best_f_cma
            best_x = best_x_cma

        # Final polish with Nelder-Mead if budget remains
        if time.time() < deadline - 10 and best_x is not None:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
