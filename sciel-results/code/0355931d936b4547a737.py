"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-stage optimizer: Halton scan, Nelder-Mead refinement, CMA-ES-style search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton(self, index, base):
        """Generate Halton sequence value."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim):
        """Generate n points in [0,1)^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # primes: 2, 3, 5, 7, 11
            for i in range(1, n + 1):
                points[i - 1, d] = self._halton(i, base)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            call_count += 1
            return func(x)

        def check_time():
            return time.time() < deadline

        # Phase 1: Halton sequence scan (30 points)
        n_halton = 30
        halton_points = self._halton_sequence(n_halton, dim)
        for i in range(n_halton):
            if not check_time() or call_count >= budget:
                break
            x = halton_points[i] * 10 - 5  # map to [-5, 5]
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead refinement from best point
        if best_x is not None and check_time() and call_count < budget:
            try:
                res = minimize(
                    lambda x: safe_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: CMA-ES style search with population
        if check_time() and call_count < budget:
            # Start from best point found
            center = best_x.copy() if best_x is not None else np.zeros(dim)
            sigma = 1.0
            pop_size = 12
            lambda_ = pop_size
            mu = pop_size // 2  # number of parents

            # Covariance matrix adaptation (simplified)
            C = np.eye(dim)
            Cs = 1.0
            step_size = 0.5

            for gen in range(20):
                if not check_time() or call_count >= budget:
                    break

                # Generate population
                pop = []
                for i in range(lambda_):
                    if call_count >= budget:
                        break
                    z = rng.standard_normal(dim)
                    # Simple CMA-ES step
                    y = center + sigma * (Cs * C @ z + (1 - Cs) * z)
                    y = np.clip(y, -5, 5)
                    pop.append((y, safe_func(y)))

                if len(pop) == 0:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[1])

                # Update best
                if pop[0][1] < best_f:
                    best_f = pop[0][1]
                    best_x = pop[0][0].copy()

                # Compute weighted mean
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                new_center = np.zeros(dim)
                for i in range(min(mu, len(pop))):
                    new_center += weights[i] * pop[i][0]

                # Update covariance (simplified)
                diff = [pop[i][0] - center for i in range(min(mu, len(pop)))]
                C_new = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    C_new += weights[i] * np.outer(diff[i], diff[i])
                C = 0.9 * C + 0.1 * C_new

                # Update step size
                old_norm = np.linalg.norm(new_center - center)
                if old_norm > 1e-10:
                    sigma *= np.exp(0.5 * (old_norm / (sigma * np.sqrt(dim)) - 1))
                sigma = np.clip(sigma, 0.01, 5.0)

                center = new_center

                # Local refinement around best
                if check_time() and call_count < budget:
                    try:
                        res = minimize(
                            lambda x: safe_func(x),
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 50, 'xatol': 1e-7, 'fatol': 1e-9}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        # Phase 4: Final local polish
        if check_time() and call_count < budget and best_x is not None:
            try:
                res = minimize(
                    lambda x: safe_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
