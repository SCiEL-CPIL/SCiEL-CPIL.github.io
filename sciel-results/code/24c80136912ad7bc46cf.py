"""Combined optimizer: CMA-ES multi-start + local refinement for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """CMA-ES with multi-start and local refinement for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin within 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def bounded_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def cma_es(x0, sigma0, max_calls, rng, deadline):
            """Run CMA-ES from x0 with initial step size sigma0, using at most max_calls evaluations."""
            dim_local = len(x0)
            x = x0.copy()
            sigma = sigma0
            fx = bounded_func(x)
            calls = 1

            # CMA-ES parameters
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim_local))))
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim_local + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim_local + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim_local) / (dim_local + 4 + 2 * mueff / dim_local)
            c_1 = 2 / ((dim_local + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim_local + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim_local) * (1 - 1.0 / (4 * dim_local) + 1.0 / (21 * dim_local ** 2))

            pc = np.zeros(dim_local)
            hs = np.zeros(dim_local)
            B = np.eye(dim_local)
            D = np.ones(dim_local)
            C = np.eye(dim_local)
            pc_norm = 0.0

            while calls < max_calls:
                if time.time() > deadline:
                    break

                # Generate lambda samples
                z = rng.standard_normal((lam, dim_local))
                # Transform using C
                y = np.array([z[i] @ B * D for i in range(lam)])
                x_pop = np.array([x + sigma * y[i] for i in range(lam)])
                x_pop = np.clip(x_pop, -5.0, 5.0)

                # Evaluate
                f_pop = np.array([bounded_func(x_pop[i]) for i in range(lam)])
                calls += lam

                # Sort
                order = np.argsort(f_pop)
                x_pop = x_pop[order]
                f_pop = f_pop[order]

                # Update mean
                x_new = np.sum(weights[:, None] * x_pop[:mu], axis=0)
                x_new = np.clip(x_new, -5.0, 5.0)

                # Update step size
                y_mean = np.sum(weights[:, None] * (x_pop[:mu] - x), axis=0) / sigma
                hs = np.sqrt((1 - (1 - c_sigma) ** 2) / (1 - c_sigma)) * np.linalg.norm(y_mean) / chi_n
                pc = (1 - c_sigma) * pc + c_sigma / hs * (y_mean if hs > 1e-20 else 0)
                pc_norm = np.linalg.norm(pc)
                hsig = (pc_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) / chi_n < 1.4 + 2 / (dim_local + 1))
                sigma *= np.exp(c_sigma / d_sigma * (pc_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                # Update covariance
                yw = np.array([(x_pop[i] - x) / sigma for i in range(mu)])
                delta_hsig = (1 - hsig) * c_sigma * (2 - c_sigma)
                C = ((1 - (c_1 + c_mu) + c_mu * delta_hsig) * C
                     + c_1 * (np.outer(pc, pc) + (1 - hsig) * delta_hsig * C)
                     + c_mu * (yw.T @ np.diag(weights) @ yw))

                # Symmetrize and recompute eigendecomposition periodically
                C = 0.5 * (C + C.T)
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = np.sqrt(eigvals)

                x = x_new
                fx = f_pop[0]

            return x, fx, calls

        # --- Phase 1: Initial exploration with structured + random points ---
        n_init = min(30, budget)
        X_init = []
        # Origin and axis-aligned points
        X_init.append(np.zeros(dim))
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 3.0
                X_init.append(pt)
        # Corner points
        for corner in range(min(8, 2 ** dim)):
            pt = np.array([(-5 if (corner >> j) & 1 else 5) for j in range(dim)])
            X_init.append(pt)
        # Random fill
        while len(X_init) < n_init:
            X_init.append(rng.uniform(-5, 5, dim))

        X_data = []
        Y_data = []
        for pt in X_init[:min(n_init, budget)]:
            if calls_made >= budget or time.time() > deadline:
                break
            x = np.clip(np.asarray(pt, dtype=float), -5.0, 5.0)
            fx = bounded_func(x)
            X_data.append(x)
            Y_data.append(fx)

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: Multi-start CMA-ES ---
        # Select diverse starting points from the initial exploration
        if len(X_data) >= 5:
            X_obs = np.array(X_data)
            Y_obs = np.array(Y_data)

            # Sort by fitness and select diverse candidates
            order = np.argsort(Y_obs)
            top_candidates = []
            min_dist = 1.5  # minimum distance between kept candidates
            for idx in order:
                if calls_made >= budget or time.time() > deadline - 10:
                    break
                diverse = True
                for _, y in top_candidates:
                    if np.linalg.norm(X_obs[idx] - y) < min_dist:
                        diverse = False
                        break
                if diverse:
                    top_candidates.append((Y_obs[idx], X_obs[idx].copy()))
                if len(top_candidates) >= 8:
                    break

            # Run CMA-ES from each candidate
            for fx0, x0 in top_candidates:
                if calls_made >= budget or time.time() > deadline - 10:
                    break

                remaining = budget - calls_made
                if remaining < 20:
                    break

                # Allocate budget: give more to better candidates
                # But ensure at least 20 calls per CMA-ES run
                alloc = min(remaining, max(20, int(remaining * 0.3)))
                sigma0 = 1.5

                x_result, f_result, used = cma_es(x0, sigma0, alloc, rng, deadline)
                if f_result < best_f:
                    best_f = f_result
                    best_x = x_result.copy()

        # --- Phase 3: Local refinement around the best point ---
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            n_refine = min(remaining, 50)
            sigma = 0.1
            for _ in range(n_refine):
                if calls_made >= budget or time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fx = bounded_func(y)
                if fx < best_f:
                    best_f = fx
                    best_x = y.copy()
                    sigma = max(0.01, sigma * 0.9)
                else:
                    sigma = min(1.0, sigma * 1.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
