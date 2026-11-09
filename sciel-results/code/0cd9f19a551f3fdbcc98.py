"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with internal restarts for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def cma_es_run(func, x0, sigma0, budget_remaining, rng, deadline, dim, calls_tracker):
            """Run CMA-ES from x0 with sigma0, using up to budget_remaining calls."""
            # Population size: lambda = 4 + floor(3 * ln(n))
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Step size control parameters
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 - cc + mueff / dim) / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            # Damping for sigma update
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            # Initial state
            xmean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            # Track calls
            calls_made = 0
            best_x = x0.copy()
            best_f = func(best_x)
            calls_made += 1
            calls_tracker[0] += 1

            gen = 0
            max_gen = 1000

            while calls_made < budget_remaining and gen < max_gen and time.time() < deadline:
                gen += 1
                # Sample population
                B = rng.multivariate_normal(np.zeros(dim), C, size=lam)
                # Normalize to unit norm for step size
                norms = np.linalg.norm(B, axis=1)
                norms[norms < 1e-10] = 1e-10
                B = B / norms[:, None]
                # Generate candidates
                Y = xmean + sigma * B
                Y = np.clip(Y, -5, 5)
                # Evaluate
                F_vals = np.array([func(y) for y in Y])
                calls_made += lam
                calls_tracker[0] += lam

                # Update best
                idx_min = np.argmin(F_vals)
                if F_vals[idx_min] < best_f:
                    best_f = F_vals[idx_min]
                    best_x = Y[idx_min].copy()

                # Sort by fitness
                order = np.argsort(F_vals)
                Y = Y[order]
                F_vals = F_vals[order]
                # Take top mu
                Ymu = Y[:mu]

                # Update mean
                xmean_old = xmean.copy()
                xmean = np.sum(weights[:, None] * Ymu, axis=0)
                xmean = np.clip(xmean, -5, 5)

                # Update covariance
                diffmean = xmean - xmean_old
                # Cumulation path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) / sigma * diffmean
                # hsig
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (gen + 1)) / chiN < 1.4 + 2 / (dim + 1))
                # Evolution path
                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * np.linalg.inv(np.sqrtm(C)) @ (diffmean / sigma)
                # Rank-mu update
                C = (1 + c1 * hsig * (hsig - 1) + cmu) * C
                C -= c1 * np.outer(pc, pc)
                C -= cmu * np.sum(weights[:, None, None] * np.outer(Ymu - xmean_old, Ymu - xmean_old) / sigma ** 2, axis=0)
                # Ensure symmetry
                C = (C + C.T) / 2
                # Eigenvalue check and repair
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
                # Update sigma
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Stagnation check: if sigma is very small or no improvement for many generations
                if sigma < 1e-6:
                    break

            return best_x, best_f, calls_made

        # Multi-start strategy: try several initial points
        # Start with a few random points, then refine the best
        num_starts = 5
        calls_tracker = [0]
        best_x_overall = None
        best_f_overall = np.inf

        # Initial random starts
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        # First pass: quick evaluation of starts
        start_fvals = np.array([func(s) for s in starts])
        calls_tracker[0] += num_starts

        # Sort starts by fitness
        start_order = np.argsort(start_fvals)
        starts = [starts[i] for i in start_order]
        start_fvals = start_fvals[start_order]

        # Allocate budget: give more to better starts
        # Use a simple strategy: try each start with a portion of budget
        remaining_budget = budget - calls_tracker[0]
        # Give the best start the most budget
        # Split remaining budget: best gets 50%, others share 50%
        if num_starts == 1:
            allocs = [remaining_budget]
        else:
            alloc_best = int(remaining_budget * 0.5)
            alloc_rest = remaining_budget - alloc_best
            alloc_others = alloc_rest // (num_starts - 1)
            allocs = [alloc_best] + [alloc_others] * (num_starts - 1)
            # Give leftover to the best
            allocs[0] += remaining_budget - sum(allocs)

        # Run CMA-ES from each start
        for i in range(num_starts):
            if calls_tracker[0] >= budget or time.time() >= deadline:
                break
            x0 = starts[i]
            sigma0 = 2.0 if i == 0 else 1.0
            b_rem = budget - calls_tracker[0]
            b_rem = min(b_rem, allocs[i])
            if b_rem < 10:
                continue
            bx, bf, cm = cma_es_run(func, x0, sigma0, b_rem, rng, deadline, dim, calls_tracker)
            if bf < best_f_overall:
                best_f_overall = bf
                best_x_overall = bx

            # If we found a very good solution, use remaining budget to polish
            if best_f_overall < 1.0:
                # Use remaining budget for a final polish run
                remaining = budget - calls_tracker[0]
                if remaining > 20:
                    bx2, bf2, cm2 = cma_es_run(func, best_x_overall, 0.1, remaining, rng, deadline, dim, calls_tracker)
                    if bf2 < best_f_overall:
                        best_f_overall = bf2
                        best_x_overall = bx2
                break

        if best_x_overall is None:
            best_x_overall = starts[0]
            best_f_overall = start_fvals[0]

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
