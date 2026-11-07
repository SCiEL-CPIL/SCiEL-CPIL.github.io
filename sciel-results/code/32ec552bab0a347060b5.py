"""Rewritten optimizer: CMA-ES with multi-start and local polish."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with local polish and (1+1)-ES fallback."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 300s limit, 30s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = np.array(x, dtype=float).copy()
                best_f = f

        # Phase 1: Multi-start CMA-ES
        # We'll run several CMA-ES instances from different starting points
        # Each gets a portion of the budget

        # Starting points: center, random, and structured
        starts = [np.zeros(dim)]
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, dim))
        for i in range(dim):
            for val in [-3, 3]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Deduplicate
        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        starts = unique_starts

        # Allocate budget: give more to the first (center) start
        # Total budget: 300 calls
        # Let's do 3 CMA-ES runs with 100 calls each, then polish

        cma_budgets = [120, 90, 60]  # total 270, leaving 30 for polish
        for i, start in enumerate(starts[:3]):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            alloc = cma_budgets[i] if i < len(cma_budgets) else 30
            alloc = min(alloc, budget - calls[0])
            if alloc < 10:
                break

            # Run a simple CMA-ES-like algorithm
            x = np.clip(start, -5, 5).copy()
            # Initial step size
            sigma = 2.0 if i == 0 else 1.5
            pop_size = 4 + int(2 * np.ceil(3 * np.log(dim)))
            pop_size = min(pop_size, max(4, alloc // 2))

            # CMA-ES state
            mean = x.copy()
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            cs = 1 / (pop_size + 4 + 2 / dim)
            c1 = 2 / ((dim + 1) ** 2 + pop_size)
            cmu = min(1 - c1, 2 / ((dim + 2) ** 2 + pop_size))
            cmu = min(cmu, 2 / ((dim + 2) ** 2 + pop_size))
            # Damping for pc
            cc = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
            # Damping for sigma
            dsigma = 1 + 2 * max(0, (dim - 1) / (dim + 1) - 1)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            best_in_cma = np.inf
            best_x_cma = mean.copy()

            for iteration in range(alloc):
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                W = np.zeros((pop_size, dim))
                for j in range(pop_size):
                    z = rng.standard_normal(dim)
                    W[j] = mean + sigma * C_sqrt @ z
                    W[j] = np.clip(W[j], -5, 5)

                # Evaluate
                F = np.zeros(pop_size)
                for j in range(pop_size):
                    try:
                        F[j] = wrapped(W[j])
                    except Exception:
                        F[j] = np.inf

                # Update best
                min_idx = np.argmin(F)
                if F[min_idx] < best_in_cma:
                    best_in_cma = F[min_idx]
                    best_x_cma = W[min_idx].copy()
                    update_best(W[min_idx], F[min_idx])

                # Sort by fitness
                order = np.argsort(F)
                W_sorted = W[order]
                F_sorted = F[order]

                # Weights
                mu_eff = pop_size // 2
                weights = np.zeros(pop_size)
                weights[:mu_eff] = np.log(mu_eff + 0.5) - np.log(np.arange(1, mu_eff + 1))
                weights /= weights.sum()

                # Update mean
                mean_old = mean.copy()
                mean = weights[:pop_size] @ W_sorted
                mean = np.clip(mean, -5, 5)

                # Update evolution paths
                delta_mean = (mean - mean_old) / (sigma * np.linalg.norm(C_sqrt))
                pc = (1 - cc) * pc + cc * delta_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (iteration + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                if hsig:
                    cs = (1 - cs) * cs + cs * (np.linalg.norm(pc) / chi_n - 1)

                # Update covariance matrix
                C_old = C.copy()
                delta = (mean - mean_old) / sigma
                rank_one = np.outer(delta, delta)
                rank_mu = np.zeros((dim, dim))
                for j in range(mu_eff):
                    y = (W_sorted[j] - mean_old) / sigma
                    rank_mu += weights[j] * np.outer(y, y)

                C = (1 + c1 * (2 - mu_eff * c1) - c1 - cmu) * C_old \
                    + c1 * rank_one + cmu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except Exception:
                    C = C_old

                # Update sigma
                sigma *= np.exp(cs / dsigma * (np.linalg.norm(pc) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10)

                # Update C_sqrt and C_inv_sqrt periodically
                if iteration % 10 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals))
                    except Exception:
                        pass

            # After CMA-ES, do a local polish from the best point found
            if best_x_cma is not None and calls[0] < budget:
                try:
                    remaining = budget - calls[0]
                    nm_maxiter = min(50, max(10, remaining // 3))
                    res = minimize(
                        wrapped,
                        best_x_cma,
                        method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
                    )
                    update_best(res.x, res.fun)
                except Exception:
                    pass

        # Phase 2: If we still have budget, do (1+1)-ES from the best point
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                if sigma < 1e-8:
                    break

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            try:
                best_f = wrapped(best_x)
            except Exception:
                best_f = 0.0

        return best_x, best_f
