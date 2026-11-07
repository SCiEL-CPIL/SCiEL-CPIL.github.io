"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock) using a portfolio approach."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        # Initial exploration: sample several starting points
        rng = np.random.default_rng(42)

        # Generate initial candidates
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Random points
        for _ in range(19):
            init_points.append(rng.uniform(-5, 5, dim))
        # Some structured points
        for val in [-3, 3]:
            init_points.append(np.full(dim, val))
        # Axis-aligned
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                init_points.append(p)

        # Evaluate initial points
        for p in init_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break
            fx = safe_call(p)
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            # Fallback: evaluate at origin
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Now run CMA-ES from the best starting point
        # CMA-ES parameters
        sigma = 1.0
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        pop_size = min(pop_size, budget - call_count - 5)

        if pop_size < 4:
            pop_size = 4

        # CMA-ES state
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = np.zeros(dim)

        # CMA-ES weights
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (2 + mueff / dim) / (dim + 2 + mueff / dim)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - cc - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Success rate adaptation for sigma
        success_count = 0
        fail_count = 0
        adaptation_window = 10

        for iteration in range(200):
            if call_count >= budget:
                break
            if time.time() > deadline - 5:
                break

            # Check if we have enough budget for a population
            if call_count + pop_size > budget:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(C))

            offspring = []
            offspring_f = []

            for i in range(pop_size):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring_f = np.array(offspring_f)
            order = np.argsort(offspring_f)
            offspring = np.array(offspring)[order]
            offspring_f = offspring_f[order]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * offspring[:mu], axis=0)

            # Update evolution paths
            y_diff = (new_mean - x_mean) / (sigma + 1e-300)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_diff
            cs = (1 - cs) * cs + np.sqrt(cs * (2 - cs) * mueff) * (
                np.sign(np.linalg.norm(pc)) * np.linalg.norm(y_diff)
            )

            # Update covariance matrix
            delta_d = (1 + 2 * max(0, cs / cs_target - 1)) if hasattr(self, 'cs_target') else 1
            cs_target = 1.0  # Simplified
            delta_d = (1 + 2 * max(0, cs / 1.0 - 1))
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * np.eye(dim)) + \
                cmu * np.sum(weights[i] * np.outer(offspring[i] - x_mean, offspring[i] - x_mean)
                             for i in range(mu)) / (sigma ** 2 + 1e-300)

            # Symmetrize C
            C = (C + C.T) / 2

            # Update sigma (success rate based)
            success_count += np.sum(offspring_f[:mu] < x_mean_fitness) if 'x_mean_fitness' in dir() else 0
            # Simplified sigma adaptation
            if np.any(offspring_f[:mu] < best_f):
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = max(1e-6, min(sigma, 10.0))

            # Update mean
            x_mean = new_mean

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-10):
                C = np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs @ eigvecs.T

        # Final local refinement using (1+1)-ES from best point
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            local_sigma = 0.5
            for _ in range(min(50, budget - call_count)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    local_sigma *= 1.3
                else:
                    local_sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
