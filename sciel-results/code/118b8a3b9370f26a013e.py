"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for Rastrigin on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda = population size
        lam = 4 + int(3 * np.log(n))  # for n=5: 4 + 3*1.609 = 8
        mu = lam // 2  # number of parents
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # sigma
        sigma = 2.0
        # step sizes for mean and covariance
        cm = (2 + 2.0 / (n + 1)) / (n + 4 + 2.0 / (n + 1))
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        # initial mean: center of box
        mean = np.zeros(n)
        # initial covariance
        C = np.eye(n)
        # evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Initial evaluation
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_used = 1

        # Precompute C^-1/2 (initially identity)
        C_inv_sqrt = np.eye(n)

        # We'll use a simplified CMA-ES with adaptive sigma
        # and restarts if stuck

        # Track best
        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate initial population around mean
        # Use a simple approach: generate lam points, evaluate, select
        # We'll do a simplified version to keep it fast

        # Actually, let's use a more efficient approach:
        # Since Rastrigin has many local minima, we need good exploration.
        # Let's use a combination of:
        # 1. Initial random sampling to find good region
        # 2. CMA-ES for fine-tuning

        # Phase 1: Random restarts to find a good basin
        # Use 20% of budget for exploration
        explore_budget = int(budget * 0.2)
        for i in range(explore_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from best point
        # Reset CMA-ES parameters
        mean = best_x.copy()
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        sigma = 1.0  # start smaller since we're in a basin

        # CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate population
            # Cholesky of C (simplified: use identity initially, update later)
            # For simplicity and speed, we'll use a diagonal approximation
            # Actually, let's do proper CMA-ES but with careful timing

            # Get C^(1/2) via Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # C is not positive definite, reset
                C = np.eye(n)
                L = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)

            # Generate lam samples
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = mean + sigma * L @ z
                offspring.append((y, i))

            # Evaluate offspring
            offspring_f = []
            for y, idx in offspring:
                if calls_used >= budget or time.time() > deadline:
                    break
                fy = evaluate(y)
                offspring_f.append((fy, y, idx))

            if len(offspring_f) < lam:
                break

            # Sort by fitness
            offspring_f.sort(key=lambda x: x[0])

            # Select mu best
            selected = [x[1] for x in offspring_f[:mu]]

            # Update mean
            m_old = mean.copy()
            mean_new = np.sum([w[i] * selected[i] for i in range(mu)], axis=0)

            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (ps @ ps.T + (1 - (1 - cs)^2) * pc @ pc.T) + cmu * sum(wi * (yi - m_old) @ (yi - m_old).T)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = 2 * (mueff - 2 + np.sum(w[:mu] ** 2) / np.sum(w) ** 2) / ((n + 2) ** 2 + mueff)
            cmu = min(cmu, 2 * (1 - c1))

            # Update ps
            C_inv_sqrt_L = np.linalg.inv(L)  # C^(-1/2)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt_L @ (mean_new - m_old) / sigma

            # Check hsig
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - explore_budget) / (n * damps))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1)

            # Update pc
            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (mean_new - m_old) / sigma
            else:
                cs = (1 - cs) * cs  # slow decay

            # Update C
            delta_sig = 1.0 if hsig else 0.0
            C = (1 - c1 - cmu + c1 * delta_sig) * C
            C += c1 * (1 - delta_sig) * ps @ ps.T + c1 * delta_sig * (ps @ ps.T + (1 - (1 - cs) ** 2) * pc @ pc.T)
            C += cmu * np.sum([w[i] * np.outer(selected[i] - m_old, selected[i] - m_old) for i in range(mu)], axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(n)

            mean = mean_new

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            sigma = max(1e-10, min(sigma, 10.0))

            # Check convergence
            if np.linalg.norm(mean_new - m_old) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
