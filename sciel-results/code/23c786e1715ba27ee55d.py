"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) with multi-restart CMA-ES."""
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with fast initial exploration for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = None
        fbest = np.inf
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, fbest
            if fx < fbest:
                fbest = fx
                best_x = x.copy()

        # Phase 1: Fast coarse search - evaluate diverse starting points
        # to get early progress quickly
        n_starts = min(15, budget // 10)
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Deterministic spread points (Sobol-like)
        for i in range(1, n_starts):
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = (np.random.uniform(-1, 1)) * 3.5
            starts.append(pt)

        # Evaluate starting points
        for s in starts:
            if not time_ok() or calls >= budget:
                break
            fx = func(s)
            calls += 1
            update_best(s, fx)

        # Phase 2: CMA-ES from the best starting point
        x0 = best_x.copy()
        sigma = 1.0  # Larger initial step size for faster exploration

        # CMA-ES parameters - tuned for dim=5
        lambda_ = 14  # Population size (slightly larger for better coverage)
        mu = lambda_ // 2  # Selection size

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def sqrtm(C):
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                return L
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-10)
                return vecs @ np.diag(np.sqrt(vals))

        B = sqrtm(C)
        gen = 0
        stall_count = 0
        fbest_prev = fbest

        while calls < budget and time_ok():
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if calls >= budget or not time_ok():
                    break
                z = np.random.standard_normal(dim)
                y = x0 + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
                fy = func(y)
                fs.append(fy)
                calls += 1

            if len(ys) < 2:
                break

            fs = np.array(fs)
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = fs[order]

            # Update best
            update_best(ys[0], fs[0])

            # Check for stall - use relative improvement threshold
            if fs[0] >= fbest_prev - max(1e-6, 0.01 * abs(fbest_prev)):
                stall_count += 1
            else:
                stall_count = 0
            fbest_prev = fbest

            # If stalled, try a restart with larger perturbation to escape local minima
            if stall_count >= 3 and calls < budget - 20:
                # Try several random restarts and pick the best
                best_restart_x = best_x
                best_restart_f = fbest
                n_restarts = min(5, (budget - calls) // 3)
                for _ in range(n_restarts):
                    if calls >= budget - 20 or not time_ok():
                        break
                    # Larger perturbation to escape Lunacek's secondary minimum
                    perturbation = 1.5 * np.random.standard_normal(dim)
                    x_restart = np.clip(best_x + perturbation, -5, 5)
                    fx_restart = func(x_restart)
                    calls += 1
                    if fx_restart < best_restart_f:
                        best_restart_f = fx_restart
                        best_restart_x = x_restart.copy()

                # Reset CMA-ES from the best restart point
                x0 = best_restart_x.copy()
                sigma = max(sigma * 0.7, 0.3)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = sqrtm(C)
                stall_count = 0
                fbest_prev = fbest
                continue

            if len(ys) < mu:
                break

            # Update mean
            x_new = sum(w[i] * ys[i] for i in range(mu))

            # Update sigma
            vals, vecs = np.linalg.eigh(C)
            vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
            C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x0) / sigma

            hsig = 1.0
            norm_ps = np.linalg.norm(ps)
            if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(c_sigma / dim * (norm_ps / np.sqrt(dim) - 1))

            # Update covariance
            pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
            yk = x_new - x0
            rank1 = np.outer(yk, yk)
            rankmu = sum(w[i] * np.outer(ys[i] - x0, ys[i] - x0) for i in range(mu))

            C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                + c_1 * (pc @ pc + (1 - (1 - c_c) ** 2) * C) \
                + c_mu * rankmu

            C = (C + C.T) / 2
            x0 = x_new
            gen += 1

            B = sqrtm(C)

        return best_x, fbest
