"""Rewritten optimizer: CMA-ES with adaptive restarts and local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with diverse restarts and final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin under 300s
        dim = self.dim
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        calls = [0]
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            calls[0] += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, budget_remaining):
            """Nelder-Mead local refinement from x0 for budget_remaining evaluations."""
            nonlocal calls, best_f, best_x
            x = np.array(x0, dtype=float)
            f0 = counted(x)

            # Build simplex
            simplex = np.empty((dim + 1, dim))
            simplex[0] = x
            fvals = np.empty(dim + 1)
            fvals[0] = f0

            for i in range(dim):
                p = x.copy()
                p[i] += 0.1
                p = np.clip(p, lo, hi)
                simplex[i + 1] = p
                fvals[i + 1] = counted(p)

            alpha, gamma, rho = 1.0, 2.0, 0.5

            while calls[0] < budget_remaining and time.time() < deadline:
                # Sort simplex by fvals
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                x_best = simplex[0]
                x_worst = simplex[-1]
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, lo, hi)
                f_r = counted(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = counted(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = counted(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            fvals[i] = counted(simplex[i])

                # Check convergence
                if np.std(fvals) < 1e-12:
                    break

            return simplex[0], fvals[0]

        def cma_es(x0, sigma0, budget_remaining, pop_size=None):
            """Run CMA-ES from x0 with initial sigma for budget_remaining evaluations."""
            nonlocal calls, best_f, best_x
            n = dim
            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(n)))
            pop_size = min(pop_size, 15)  # cap for 5D

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Track best
            local_best_x = mean.copy()
            local_best_f = counted(local_best_x)

            # Generation loop
            while calls[0] < budget_remaining and time.time() < deadline:
                # Generate population
                pops = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    # Apply C matrix via Cholesky (approximate)
                    # For 5D, we can do full Cholesky
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    y = L @ z
                    x_new = mean + sigma * y
                    x_new = np.clip(x_new, lo, hi)
                    pops.append((x_new, counted(x_new)))

                # Sort by fitness
                pops.sort(key=lambda t: t[1])

                # Update best
                if pops[0][1] < local_best_f:
                    local_best_f = pops[0][1]
                    local_best_x = pops[0][0].copy()
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Select
                selected = [p[0] for p in pops[:mu]]

                # Update mean
                new_mean = sum(w * x for w, x in zip(weights, selected))
                new_mean = np.clip(new_mean, lo, hi)

                # Update sigma (CSA - Cumulative Step Adaptation)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (new_mean - mean))
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls[0] // pop_size))) < (1 + 2 / (n + 1)) * chi_n else 0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma

                # Update C (rank-mu and rank-one update)
                delta = hsig * cc * (2 - cc)
                C = (1 + c1 * delta - c1) * C + c1 * np.outer(pc, pc)
                C += cmu * sum(w * np.outer(y, y) for w, y in zip(weights, [(x - mean) / sigma for x in selected]))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(n)

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                mean = new_mean

                # Check convergence
                if np.linalg.norm(mean - local_best_x) < 1e-6 * sigma:
                    break
                if sigma < 1e-8:
                    break

            return local_best_x, local_best_f

        # Phase 0: Random sampling to find diverse, promising starting points
        n_random = min(15, self.budget // 20)
        candidates = []
        for _ in range(n_random):
            if calls[0] >= self.budget or time.time() >= deadline:
                break
            x = rng.uniform(lo, hi, dim)
            fx = counted(x)
            candidates.append((x, fx))

        # Sort by fitness, keep the best as CMA-ES seed
        candidates.sort(key=lambda t: t[1])
        seed_x = candidates[0][0]
        seed_f = candidates[0][1]

        # Phase 1: CMA-ES with moderate budget
        cma_budget = int(self.budget * 0.7)
        if calls[0] < cma_budget and time.time() < deadline:
            cma_es(seed_x, 2.0, cma_budget)

        # Phase 2: If we have budget left, do a second CMA-ES from a different random point
        if calls[0] < int(self.budget * 0.9) and time.time() < deadline:
            # Pick a second seed from candidates (if different from first)
            if len(candidates) > 1:
                seed2 = candidates[1][0]
            else:
                seed2 = rng.uniform(lo, hi, dim)
            remaining = int(self.budget * 0.9) - calls[0]
            if remaining > 20:
                cma_es(seed2, 1.5, remaining)

        # Phase 3: Local refinement with Nelder-Mead from best point
        if calls[0] < self.budget and time.time() < deadline:
            refine_budget = self.budget - calls[0]
            if best_x is not None:
                nelder_mead(best_x, refine_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
