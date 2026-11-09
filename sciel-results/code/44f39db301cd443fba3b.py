"""Rewritten optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for BBOB f7 (Step ellipsoidal) in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        total_calls = 0

        def call_func(x):
            nonlocal total_calls
            total_calls += 1
            return func(x)

        # Global best
        best_x = np.zeros(dim)
        best_f = np.inf

        rng = np.random.default_rng(42)

        def run_cma_es(x0, sigma0, remaining_budget, deadline):
            nonlocal best_x, best_f, total_calls

            if remaining_budget < 2 * dim:
                return best_x, best_f

            dim = self.dim
            mu = max(1, dim // 2)
            lam = 2 * mu
            if lam < 4:
                lam = 4
                mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = int(np.sum(w[:int(1.0 / np.sum(w ** 2))]) ** 2)
            mueff = max(1, min(mueff, mu))
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - dim - 1) / (dim + mueff - 3)) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 - mu_eff_c1 + mueff * c_sigma) / ((dim + 1.3) ** 2 + mueff * c_sigma)
            c_1 = (2 - c_sigma + mueff * c_sigma) / ((dim + 1.3) ** 2 + mueff * c_sigma)
            c_mu = min(1 - c_1, (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Precompute Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            for gen in range(10000):
                if total_calls >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    noise = rng.standard_normal(dim)
                    x_child = x_mean + sigma * (C_sqrt @ noise)
                    x_child = np.clip(x_child, -5, 5)
                    f_child = call_func(x_child)
                    offspring.append((f_child, x_child))
                    if total_calls >= budget:
                        break

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update global best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Select top mu
                selected = [o[1] for o in offspring[:mu]]
                x_new_mean = np.sum(w[:, None] * np.array(selected), axis=0)

                # Update C matrix
                y = (x_new_mean - x_mean) / sigma
                pc = (1 - c_c) * pc + c_c * y
                norm_pc = np.linalg.norm(pc)
                h_sig = 1 if (norm_pc / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1))) / np.sqrt(1 - (1 - c_c) ** 2)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                ps = (1 - c_sigma) * ps + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mueff) / norm_pc * y if norm_pc > 1e-16 else (1 - c_sigma) * ps
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu * np.sum(w[:, None, None] * np.outer(selected[i] - x_mean, selected[i] - x_mean) / sigma ** 2 for i in range(mu))

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update mean
                x_mean = x_new_mean

                # Update Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-10 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_sqrt = np.eye(dim)

                # Check convergence
                if np.all(np.abs(eigvals) < 1e-12) or sigma < 1e-15:
                    break

            return best_x, best_f

        # Phase 1: Initial exploration to find a good starting point
        n_init = min(30, budget // 5)
        candidates = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        candidates.sort(key=lambda t: t[0])
        x0 = candidates[0][1].copy()
        f0 = candidates[0][0]

        # Estimate initial sigma
        inits = np.array([c[1] for c in candidates[:min(10, len(candidates))]])
        f_inits = np.array([c[0] for c in candidates[:min(10, len(candidates))]])
        if len(inits) > 2 and np.ptp(f_inits) > 1e-10:
            scale_est = np.std(inits, axis=0)
            sigma0 = np.mean(scale_est) * 0.5
            sigma0 = np.clip(sigma0, 0.1, 2.0)
        else:
            sigma0 = 1.0

        # Phase 2: CMA-ES with restarts
        remaining = budget - total_calls
        restart_count = 0
        max_restarts = 5
        stall_threshold = 50

        while remaining > 2 * dim and restart_count < max_restarts and time.time() < deadline - 5.0:
            f_before = best_f
            x0, sigma0 = run_cma_es(x0, sigma0, remaining, deadline)
            remaining = budget - total_calls

            if best_f < f_before - 1e-8:
                x0 = best_x.copy()
                sigma0 = sigma0 * 0.5 if sigma0 > 0.01 else sigma0
            else:
                restart_count += 1
                # Perturb for restart
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)
                sigma0 = max(sigma0, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
