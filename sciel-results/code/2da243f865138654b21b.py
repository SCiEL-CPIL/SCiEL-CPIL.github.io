"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island model with 1+1-CMA-ES for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diversified initial sampling
        n_init = min(30, budget // 8)
        # Structured points
        structured = [
            np.zeros(n),
            np.ones(n) * 3.0,
            -np.ones(n) * 3.0,
            np.array([3, -3, 3, -3, 3])[:n],
            np.array([-3, 3, -3, 3, -3])[:n],
        ]
        for p in structured:
            if total_calls >= budget:
                break
            try_eval(p)

        # Random samples
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            scale = 3.5 * (1.0 - 0.5 * i / max(1, n_init))
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        # Phase 2: Island model with 1+1-CMA-ES
        n_islands = 4
        # Initialize islands from best points found
        # We'll use the best point and random perturbations for diversity
        islands = []
        if best_x is not None:
            islands.append(best_x.copy())
        for _ in range(n_islands - 1):
            x_try = rng.uniform(-4, 4, n)
            f_try = try_eval(x_try)
            if f_try is not None:
                islands.append(x_try.copy())
            else:
                islands.append(best_x.copy() if best_x is not None else np.zeros(n))

        def cma_11(x0, sigma0, max_calls, rng_seed):
            """1+1-CMA-ES with covariance adaptation."""
            nonlocal total_calls, best_x, best_f
            local_rng = np.random.default_rng(rng_seed)

            x = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)

            # 1+1-CMA-ES parameters
            cs = (n + 2) / (n + 4)
            c1 = 2 / ((n + 1.3) ** 2 + 2)
            cmu = min(1 - c1, (n - 2 + 1 / n) / ((n + 2) ** 2 + 2))
            cc = (4 + (n + 2) / n) / (n + 4 + 2 * (n + 2) / n)
            hsig = (1 - (1 - cc) ** (1 / cs)) / np.sqrt(1 - (1 - cc) ** (2 / cs))

            best_x_local = x0.copy()
            best_f_local = best_f

            while total_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Generate one trial point
                z = local_rng.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break

                if fy < best_f_local:
                    best_f_local = fy
                    best_x_local = y.copy()

                # Compute C_sqrt_inv
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-14)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                d = y - x
                C_sqrt_inv_d = C_sqrt_inv @ d
                norm_d = np.linalg.norm(C_sqrt_inv_d)

                # Update sigma
                if norm_d > 1e-12:
                    p_sigma = (1 - cs) * p_sigma + cs * (norm_d / np.sqrt(2) - 1) * C_sqrt_inv_d / norm_d
                    sigma *= np.exp(cs / (n + 2) * (norm_d / np.sqrt(2) - 1))
                    sigma = np.clip(sigma, 1e-12, 5.0)

                # Update C (rank-1)
                C = (1 + c1 * cc) * C + c1 * np.outer(d, d)

                # Update p_c
                if norm_d > 1e-12:
                    p_c = (1 - cc) * p_c + cc * hsig * C_sqrt_inv_d / np.sqrt(2)
                C = (1 - c1 * cc * (2 - cc)) * C + c1 * (1 + cc * (2 - cc)) * np.outer(p_c, p_c)

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T

                # Update mean
                x = y
                x = np.clip(x, -5, 5)

                if sigma < 1e-8:
                    break

            return best_x_local, best_f_local

        # Run 1+1-CMA-ES on each island with staggered budgets
        calls_per_island = max(30, (budget - total_calls) // n_islands)
        for i in range(n_islands):
            if total_calls >= budget - 10 or time.time() > deadline - 30:
                break
            remaining = min(calls_per_island, budget - total_calls - 5)
            if remaining < 10:
                break
            x_opt, f_opt = cma_11(islands[i], 0.8, remaining, rng_seed=1000 + i * 7)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: Final polish with coordinate descent
        if total_calls < budget and best_x is not None and time.time() < deadline - 10:
            step = 0.1
            for _ in range(4):
                if total_calls >= budget or time.time() > deadline - 5:
                    break
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 5:
                        break
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            break
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                step *= 0.5

            # Final small perturbations
            n_final = min(15, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.02, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
