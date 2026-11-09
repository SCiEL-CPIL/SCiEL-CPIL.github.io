"""CMA-ES optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4 for 5D Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        d = self.dim
        n = 4  # Population size

        # Initial mean: center of the box
        m = np.zeros(d)
        # Initial sigma: cover the box
        sigma = 3.0

        # Initial covariance matrix
        C = np.eye(d)
        # Inverse covariance
        C_inv = np.eye(d)
        # Cholesky factor
        L = np.eye(d)
        # Square root of inverse covariance
        C_sqrt_inv = np.eye(d)

        # Learning rates
        cc = (4 + d / n) / ((d + 4) ** 2 + n)
        cs = (2 + 2 / (d + 1)) / ((d + 1) ** 2 + 4 + 2 / (d + 1))
        cmu = min(1 - cs, 2 * (d - 2 + 1 / d) / ((d + 2) ** 2 + n))
        c1 = 2 / ((d + 1.3) ** 2 + n)
        cmu = min(1 - c1 - cs, cmu)

        # Step size control
        damps = 1 + 2 * max(0, np.sqrt((n - 1) / (d + 1)) - 1) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        hsig = 0

        best_x = m.copy()
        best_f = func(best_x)
        calls = 1

        # Precompute constants
        while calls < self.budget:
            if time.time() > deadline:
                break

            # Sample points
            ys = []
            ws = []
            for i in range(n):
                # Standard normal
                z = rng.standard_normal(d)
                # Transform
                y = C_sqrt_inv @ z
                ys.append(y)
                ws.append(1.0 / n)

            # Weights for recombination (standard CMA-ES uses mu = n/2, but for small n we use all)
            # For n=4, mu=2 is standard. Let's use mu=2 to be efficient and standard.
            mu = n // 2
            weights = np.zeros(mu)
            weights[:mu] = 1.0 / mu
            # Normalize weights
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Generate population
            pop = []
            for i in range(n):
                z = rng.standard_normal(d)
                x_i = m + sigma * (C_sqrt_inv @ z)
                x_i = np.clip(x_i, -5, 5)
                pop.append(x_i)

            # Evaluate
            fvals = []
            for x_i in pop:
                fvals.append(func(x_i))
                calls += 1
                if time.time() > deadline:
                    break

            if len(fvals) < n:
                break

            # Sort
            idx = np.argsort(fvals)
            pop_sorted = [pop[i] for i in idx]
            fvals_sorted = [fvals[i] for i in idx]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Select best mu
            xw = np.zeros(d)
            for i in range(mu):
                xw += weights[i] * pop_sorted[i]

            # Update mean
            m_old = m.copy()
            m = xw

            # Update C and sigma
            # Compute step
            step = (m - m_old) / sigma

            # Success history rule
            hsig = 1.0 if (step @ step) < (1.7 + 2.0 / (d + 1)) * chiN ** 2 else 0.0

            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * step

            # Check path length
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (calls - n))) > (1.4 + 1 / d) * chiN:
                # Adapt C
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((d, d))
                for i in range(mu):
                    zi = (pop_sorted[i] - m_old) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * hsig * (2 - cc) - c1 - cs) * C + c1 * rank_one + cs * (rank_mu - C)
                # Ensure symmetry
                C = (C + C.T) / 2
            else:
                C = (1 - c1 - cs) * C + c1 * np.outer(pc, pc) + cs * (np.outer(pc, pc) - C)
                # Wait, standard update is:
                # C = (1 + c1*(2-cc) - c1 - cs)*C + c1*pc*pc^T + cs*(sum wi zi zi^T - C)
                # If hsig=0, the first term is (1 - c1 - cs)*C + c1*pc*pc^T + cs*(rank_mu - C)
                # Let's just use the general formula with hsig
                C = (1 + c1 * hsig * (2 - cc) - c1 - cs) * C + c1 * np.outer(pc, pc)
                if hsig == 0:
                    # We need rank_mu even if hsig is 0? No, the standard update is:
                    # C_{t+1} = (1 + c_sigma(2-c_c) - c_sigma - c_1) C_t + c_1 pc pc^T + c_sigma (sum wi zi zi^T - C_t)
                    # Wait, the c_sigma term is always there.
                    pass # The formula above handles it if we compute rank_mu always

            # Recompute rank_mu for the C update if not done or to be safe
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                zi = (pop_sorted[i] - m_old) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            # Correct C update
            C = (1 + c1 * hsig * (2 - cc) - c1 - cs) * C + c1 * np.outer(pc, pc) + cs * (rank_mu - C)
            C = (C + C.T) / 2

            # Update sigma
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * step
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or add jitter
                C = C + 1e-6 * np.eye(d)
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)

        return best_x, best_f
# EVOLVE-BLOCK-END
