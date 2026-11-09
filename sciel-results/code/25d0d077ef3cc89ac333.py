"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Rastrigin on [-5,5]^5 with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_left():
            return deadline - time.time()

        # Phase 1: Random exploration to find good basin
        explore_budget = min(50, int(budget * 0.1))
        best_x = None
        best_f = np.inf
        calls_used = 0

        for i in range(explore_budget):
            if time_left() < 5:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            calls_used += 1
            best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        remaining = budget - calls_used

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))  # for n=5: 8
        mu = lam // 2  # 4
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + np.sum(w[:mu] ** 2) / np.sum(w) ** 2) / ((n + 2) ** 2 + mueff)
        cmu = min(cmu, 2 * (1 - c1))
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        # Run CMA-ES with adaptive restarts
        # Start with a moderate sigma
        max_restarts = 5
        restart_threshold = 1e-6

        for restart in range(max_restarts):
            if calls_used >= budget or time_left() < 10:
                break

            # Initialize CMA-ES state
            mean = global_best_x.copy()
            sigma = 1.0 / (restart + 1)  # Decreasing sigma for restarts
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            gen = 0

            # Track convergence
            last_mean = mean.copy()
            stagnant = 0

            while calls_used < budget and time_left() > 5:
                gen += 1

                # Generate population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

                # Generate lam samples
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = mean + sigma * L @ z
                    offspring.append(y)

                # Evaluate offspring
                offspring_f = []
                for y in offspring:
                    if calls_used >= budget or time_left() < 5:
                        break
                    y_clipped = np.clip(y, -5, 5)
                    fy = func(y_clipped)
                    calls_used += 1
                    offspring_f.append((fy, y_clipped))
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y_clipped.copy()

                if len(offspring_f) < lam:
                    break

                # Sort by fitness
                offspring_f.sort(key=lambda x: x[0])

                # Select mu best
                selected = [x[1] for x in offspring_f[:mu]]
                selected_y = [x[1] for x in offspring_f[:mu]]

                # Update mean
                m_old = mean.copy()
                mean_new = np.sum([w[i] * selected[i] for i in range(mu)], axis=0)

                # Update evolution paths
                # C^(-1/2) = inv(L)
                try:
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mean_new - m_old) / sigma

                # Check hsig
                chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n ** 2))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (n + 1)) * chi_n

                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (mean_new - m_old) / sigma

                # Update covariance matrix
                delta_sig = 1.0 if hsig else 0.0
                C = (1 - c1 - cmu) * C
                C += c1 * (1 - delta_sig) * ps @ ps.T + c1 * delta_sig * (ps @ ps.T + (1 - (1 - cs) ** 2) * pc @ pc.T)
                C += cmu * np.sum([w[i] * np.outer(selected_y[i] - m_old, selected_y[i] - m_old) for i in range(mu)], axis=0)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(n)

                mean = mean_new

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 10.0))

                # Check convergence
                if np.linalg.norm(mean_new - m_old) < 1e-8:
                    stagnant += 1
                else:
                    stagnant = 0

                if stagnant > 10:
                    break

            # If we converged or used up budget, stop
            if calls_used >= budget or time_left() < 10:
                break

            # Otherwise, restart from global best with smaller sigma
            # Only restart if we made progress
            if global_best_f >= best_f * 0.95:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
