"""Optimized BBOB f9 (Rosenbrock rotated) solver with CMA-ES + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """CMA-ES global search + L-BFGS-B refinement for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        # Wrap func to count calls and ensure we stay within budget
        state = {'count': 0, 'best_x': None, 'best_f': np.inf}

        def counted_func(x):
            state['count'] += 1
            if state['count'] > budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: CMA-ES for global exploration
        # CMA-ES adapts to the anisotropic landscape of rotated Rosenbrock
        cma_budget = int(budget * 0.6)  # Use 60% of budget for CMA-ES
        cma_calls = 0
        cma_deadline = deadline - 60  # Reserve time for L-BFGS-B

        # Initialize CMA-ES manually
        x0 = np.zeros(dim)
        sigma0 = 3.0  # Initial step size
        popsize = max(4, int(4 + 3 * np.log(dim)))  # Population size
        cma_calls += 1  # Initial evaluation
        f0 = counted_func(x0)
        best_x = x0.copy()
        best_f = f0
        total_calls = state['count']

        # CMA-ES state
        mean = x0.copy()
        sigma = sigma0
        B = np.eye(dim)  # Rotation matrix
        D = np.ones(dim)  # Axis lengths

        # CMA-ES parameters
        cc = (4 + dim / 3) / (dim + 4 + 2 / dim)
        cmu = min(0.3, (2 * (dim - 2) + 1) / ((dim + 2) ** 2 + 1))
        c1 = 2 / ((dim + 1.3) ** 2 + mu)
        cs = (mu + 2) / (dim + mu + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        ps = np.zeros(dim)
        hsig = (1 + 2 / (dim + 1)) * chiN

        archive = [(f0, x0.copy())]

        for gen in range(200):
            if time.time() > cma_deadline:
                break
            if total_calls >= cma_budget:
                break

            # Generate population
            for i in range(popsize):
                if total_calls >= cma_budget:
                    break
                if time.time() > cma_deadline:
                    break

                zi = rng.standard_normal(dim)
                xi = mean + sigma * (B @ (D * zi))
                fi = counted_func(xi)
                total_calls = state['count']

                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

                archive.append((fi, xi.copy()))

            if len(archive) < mu + 1:
                break

            # Sort by fitness
            archive.sort(key=lambda t: t[0])

            # Update mean
            new_mean = np.sum([weights[i] * archive[i][1] for i in range(mu)], axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chiN * (B @ (D * (new_mean - mean) / sigma))
            if np.linalg.norm(ps) < (1.4 + 1 / dim) * hsig:
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))

            # Update covariance matrix
            if np.linalg.norm(ps) <= hsig:
                h1 = 1
            else:
                h1 = 0

            rank_one = np.outer(new_mean - mean, new_mean - mean)
            diff = [archive[i][1] - mean for i in range(mu)]
            rank_mu = np.sum([weights[i] * np.outer(diff[i], diff[i]) for i in range(mu)], axis=0)

            C = (1 + c1 * h1 * (1 - h1) - cmu) * B @ np.diag(D ** 2) @ B.T
            C += c1 * (1 + h1 * (1 - h1) - c1) * rank_one + cmu * rank_mu

            # Eigendecomposition for B and D
            eigenvalues, B_new = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            D_new = np.sqrt(eigenvalues)

            mean = new_mean
            B = B_new
            D = D_new

        # Phase 2: L-BFGS-B refinement from best CMA-ES point
        if best_x is not None and total_calls < budget - 20 and time.time() < deadline:
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': max(5, budget - total_calls - 5),
                             'ftol': 1e-12, 'gtol': 1e-8}
                )
                total_calls = state['count']
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except (ValueError, Exception):
                pass

        # Phase 3: High-precision polish
        if best_x is not None and total_calls < budget - 10 and time.time() < deadline:
            try:
                result2 = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': max(5, budget - total_calls - 5),
                             'ftol': 1e-15, 'gtol': 1e-10}
                )
                total_calls = state['count']
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5, 5).copy()
            except (ValueError, Exception):
                pass

        # Final: ensure best_x is set
        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
