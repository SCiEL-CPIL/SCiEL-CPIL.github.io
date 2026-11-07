"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Global random sampling + CMA-ES optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global random sampling to find a good initial point
        n_global = 100
        if budget <= n_global:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            for _ in range(budget - 1):
                y = rng.uniform(-5, 5, dim)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx

        best_x = None
        best_f = np.inf
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        remaining = budget - n_global
        global_best_x = best_x.copy()
        global_best_f = best_f

        # CMA-ES parameters
        popsize = max(8, 4 + int(3 * np.log(dim + 1)))
        popsize = min(popsize, max(4, remaining // 5))

        def cma_es(x_init, fx_init, calls_budget, sigma_init=2.0):
            """Run CMA-ES with given budget."""
            x = x_init.copy()
            fx = fx_init
            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            cs = (popsize + 2) / (dim + 4)
            ds = 1 + 2 * max(0, (np.sqrt((popsize + 1) / (dim + 3)) - 1) / (dim + 1)) + cs
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + popsize)
            cmu = min(1 - c1, 2 * (popsize - 2 + 1.0 / popsize) / ((dim + 2) ** 2 + popsize))
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Weights
            w = np.exp(-5 * np.arange(popsize) / popsize)
            w /= w.sum()
            mu_eff = (w ** 2).sum() ** -1

            local_best_x = x.copy()
            local_best_f = fx
            calls_left = calls_budget
            stall_counter = 0
            no_improve_threshold = 20  # generations without improvement

            while calls_left >= popsize:
                # Generate population
                z = rng.standard_normal((popsize, dim))
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                candidates = np.array([
                    np.clip(x + sigma * (L @ z[i]), -5, 5) for i in range(popsize)
                ])

                # Evaluate
                fvals = np.array([func(c) for c in candidates])
                calls_left -= popsize

                # Update local best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < local_best_f:
                    local_best_f = fvals[idx_best]
                    local_best_x = candidates[idx_best].copy()
                    stall_counter = 0
                else:
                    stall_counter += 1

                # Check for stall and potentially restart
                if stall_counter >= no_improve_threshold and sigma < 0.01:
                    break

                # Sort
                order = np.argsort(fvals)
                fvals_sorted = fvals[order]
                x_sorted = candidates[order]

                # Save old mean
                x_old = x.copy()

                # Update mean
                x = np.sum(w[:, None] * x_sorted, axis=0)

                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi * (L.T @ (x - x_old))

                # Update pc
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * (x - x_old) / sigma

                # Update C
                delta_hs = (np.sum(ps ** 2) / (1 - (1 - cs) ** 2)) / (dim * (1 - (1 - cs) ** 2)) - 1
                hsig = 1.0 if delta_hs < (1 + 4 / (dim + 1)) else 0.0

                C = (1 + c1 * hsig + cmu - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) - (1 - hsig) * C)
                C += cmu * np.sum(w[:, None, None] * np.outer(x_sorted - x_old, x_sorted - x_old), axis=0) / sigma ** 2

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)

            # Use remaining calls
            while calls_left > 0:
                n_gen = min(popsize, calls_left)
                z = rng.standard_normal((n_gen, dim))
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                candidates = np.array([
                    np.clip(x + sigma * (L @ z[i]), -5, 5) for i in range(n_gen)
                ])
                fvals = np.array([func(c) for c in candidates])
                calls_left -= n_gen

                idx_best = np.argmin(fvals)
                if fvals[idx_best] < local_best_f:
                    local_best_f = fvals[idx_best]
                    local_best_x = candidates[idx_best].copy()

            return local_best_x, local_best_f

        # Run CMA-ES in phases
        phase1_budget = min(remaining // 2, 200)
        x1, f1 = cma_es(best_x, best_f, phase1_budget, sigma_init=2.0)

        if f1 < global_best_f:
            global_best_f = f1
            global_best_x = x1.copy()

        remaining2 = remaining - phase1_budget

        if remaining2 > 20:
            phase2_budget = min(remaining2, 150)
            x2, f2 = cma_es(x1, f1, phase2_budget, sigma_init=0.5)

            if f2 < global_best_f:
                global_best_f = f2
                global_best_x = x2.copy()

            remaining3 = remaining2 - phase2_budget

            if remaining3 > 10:
                phase3_budget = remaining3
                x3, f3 = cma_es(x2, f2, phase3_budget, sigma_init=0.1)

                if f3 < global_best_f:
                    global_best_f = f3
                    global_best_x = x3.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
