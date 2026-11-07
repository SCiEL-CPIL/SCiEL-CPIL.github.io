"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and LHS exploration for multimodal optimization on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma, n_evals, max_stagnation=25):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            n = dim
            lam = max(6, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            xmean = x0.copy()
            C = np.eye(n)
            # Compute initial Cinv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-12))
            B = eigvecs
            Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            evals_used = 0
            no_improve = 0
            local_best = best_f
            sigma_min = 1e-10
            sigma_max = 10.0

            for gen in range(n_evals // lam + 10):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Check if sigma is too small or too large
                if sigma < sigma_min or sigma > sigma_max:
                    break

                # Generate population
                ys = []
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = xmean + sigma * Cinv_sqrt @ rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    ys.append((y, fy))

                if len(ys) < 2:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[1])

                # Check for improvement
                if ys[0][1] < local_best - 1e-10:
                    local_best = ys[0][1]
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart if stagnating
                if no_improve > max_stagnation:
                    break

                # Update mean
                xnew = np.zeros(n)
                for i in range(mu):
                    xnew += weights[i] * ys[i][0]
                xmean = xnew

                # Update step size control variable
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * Cinv_sqrt @ (xnew - xmean)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used + 1) / n)) < (1.4 + 2 / (n + 1)) * chi_n)

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * Cinv_sqrt @ (xmean - xnew)
                xmean = xnew  # this is the new mean

                # Update covariance matrix
                delta = 1.0 if hsig else 0.0
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = Cinv_sqrt @ (ys[i][0] - xmean)
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Ensure PSD
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                # Update D and B
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, sigma_min)
                sigma = min(sigma, sigma_max)

                evals_used += len(ys)

                if evals_used >= n_evals:
                    break

        # Phase 1: Latin Hypercube Sampling for initial exploration
        # Use ~25% of budget for exploration
        n_explore = max(15, int(budget * 0.25))

        def lhs_sample(n, dim, rng):
            """Generate n points using Latin Hypercube Sampling in [-5, 5]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.uniform(size=n)) / n * 10 - 5
            return points

        lhs_points = lhs_sample(n_explore, dim, rng)
        lhs_values = []

        for x in lhs_points:
            if time.time() > deadline or calls_made >= budget:
                break
            fx = call_func(x)
            lhs_values.append(fx)

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            try:
                fx = call_func(x)
                return x, fx
            except:
                return np.zeros(dim), float('inf')

        # Find the best few points from LHS for multi-start
        if len(lhs_values) > 0:
            sorted_indices = np.argsort(lhs_values)
            n_starts = min(4, len(sorted_indices))
            start_points = [lhs_points[idx].copy() for idx in sorted_indices[:n_starts]]
        else:
            start_points = [best_x.copy()]

        # Phase 2: Multiple short CMA-ES runs from different starting points
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            per_start = max(10, remaining // (n_starts + 1))

            for s in range(n_starts):
                if calls_made >= budget or time.time() > deadline:
                    break

                if s == 0:
                    x0 = start_points[0]
                    sigma0 = 1.5
                elif s == 1 and len(start_points) > 1:
                    x0 = start_points[1]
                    sigma0 = 1.0
                else:
                    # Start near current best with some perturbation
                    if best_x is not None:
                        x0 = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
                    else:
                        x0 = rng.uniform(-5, 5, dim)
                    sigma0 = 1.0

                cma_es(x0, sigma0, per_start)

        # Phase 3: One longer CMA-ES run from the best point found
        if calls_made < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                cma_es(best_x.copy(), 0.5, remaining, max_stagnation=15)

        # Phase 4: Local refinement with (1+1)-ES if budget remains
        if calls_made < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
