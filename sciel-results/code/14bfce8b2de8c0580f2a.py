"""Optimizer for BBOB f17 (Schaffer's F7) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, leave 10s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        for _ in range(10):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: CMA-ES
        def cma_es(x0, sigma0, max_calls, rng_local):
            nonlocal best_x, best_fx, calls_made

            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (1 - cc - 2 * (min(1, cs - 1) * (n - 1) / n))
            cmu = (2 / ((n + 2) ** 2 + mueff)) * (mueff - 2 + 2 / (n - 2)) - c1
            cmu = min(cmu, 1 - c1)

            sigma = sigma0
            mean = x0.copy()
            C = np.eye(n)
            C_eig = np.linalg.eigh(C)
            C_eigvals = C_eig[0]
            C_eigvecs = C_eig[1]
            C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals)) @ C_eigvecs.T

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial population
            pop = []
            fvals = []
            for i in range(lam):
                z = rng_local.standard_normal(n)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                pop.append(x)
                fvals.append(fx)

            # Update best
            idx = np.argmin(fvals)
            if fvals[idx] < best_fx:
                best_fx = fvals[idx]
                best_x = pop[idx].copy()

            # Sort by function value
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            max_iter = 100
            for iteration in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Update mean
                old_mean = mean.copy()
                mean = sum(w[i] * pop[i] for i in range(mu))

                # Update evolution paths
                if np.linalg.norm(C_sqrt.T @ (mean - old_mean) / sigma) > (1.4 + 1 / cc) * cs * chi_n:
                    pc = (1 - cs) * pc
                else:
                    pc = (1 - cs) * pc + cs / chi_n * (C_sqrt.T @ (mean - old_mean)) / sigma

                hsig = (np.sqrt((1 - (1 - cc) ** (2 * (iteration + 1))) / (1 - (1 - cc) ** 2)) - 1 + cc) * chi_n
                if np.linalg.norm(pc) / chi_n < (1.4 + 1 / cc) * hsig:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (mean - old_mean) / sigma
                    delta_sig = 1 + cs * (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1)
                else:
                    ps = (1 - cs) * ps

                # Update covariance
                delta_heav = 1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1) / (n + 3)) * (cmu - c1)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y = (pop[i] - old_mean) / sigma
                    rank_mu += w[i] * np.outer(y, y)

                C = (1 - cs * c1 - cmu) * C + (1 - cs) * c1 * np.outer(pc, pc) + cmu * rank_mu

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    C_eig = np.linalg.eigh(C)
                    C_eigvals = np.maximum(C_eig[0], 1e-12)
                    C_eigvecs = C_eig[1]
                    C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
                    C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Check convergence
                if np.linalg.norm(mean - old_mean) < 1e-10 * max(1, sigma):
                    break

                # Generate new population
                new_pop = []
                new_fvals = []
                for i in range(lam):
                    z = rng_local.standard_normal(n)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    fx = call_func(x)
                    new_pop.append(x)
                    new_fvals.append(fx)

                # Update best
                idx = np.argmin(new_fvals)
                if new_fvals[idx] < best_fx:
                    best_fx = new_fvals[idx]
                    best_x = new_pop[idx].copy()

                # Sort
                order = np.argsort(new_fvals)
                pop = [new_pop[i] for i in order]
                fvals = [new_fvals[i] for i in order]

            return best_x, best_fx

        # Main CMA-ES run
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            first_run_calls = min(remaining, 350)
            x0 = best_x.copy()
            _, _ = cma_es(x0, 1.0, calls_made + first_run_calls, rng)

        # Second run if budget allows
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                x0_2 = best_x + rng.normal(0, 0.5, d)
                x0_2 = np.clip(x0_2, -5, 5)
                fx2 = call_func(x0_2)
                if fx2 < best_fx:
                    best_fx = fx2
                    best_x = x0_2.copy()

                if calls_made < budget and time.time() < deadline:
                    second_run_calls = min(budget - calls_made, 40)
                    _, _ = cma_es(best_x.copy(), 0.5, calls_made + second_run_calls, rng)

        # Phase 3: Final polish
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(15, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
