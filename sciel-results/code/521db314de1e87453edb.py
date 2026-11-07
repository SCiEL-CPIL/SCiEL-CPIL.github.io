"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive population for multimodal f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        n = self.dim
        budget = self.budget

        def safe_call(x):
            try:
                val = func(np.asarray(x, dtype=float))
                return float(val)
            except Exception:
                return float('inf')

        # Phase 1: Deterministic multi-start grid for early progress
        starts = []
        starts.append(np.zeros(n))
        for mask in range(2 ** n):
            pt = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            starts.append(pt)
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 2.5
                starts.append(pt)
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 4.0
                starts.append(pt)
        # Add some intermediate points
        for i in range(n):
            for j in range(i + 1, n):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        pt = np.zeros(n)
                        pt[i] = si * 2.0
                        pt[j] = sj * 2.0
                        starts.append(pt)

        unique_starts = []
        seen = set()
        for pt in starts:
            key = tuple(pt)
            if key not in seen:
                seen.add(key)
                unique_starts.append(pt)

        max_starts = min(len(unique_starts), budget // 3)
        best_x = unique_starts[0].copy()
        best_f = safe_call(best_x)
        calls = 1

        for i in range(1, max_starts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = unique_starts[i]
            f = safe_call(pt)
            calls += 1
            if f < best_f:
                best_x, best_f = pt.copy(), f

        # Phase 2: CMA-ES from best point
        x = best_x.copy()
        sigma = 1.0
        pop_size = max(4, int(4 + 3 * np.log(n)))
        
        # Initialize covariance matrix as identity
        C = np.eye(n)
        C_inv = np.eye(n)
        
        # CMA-ES parameters
        alpha_mu = 1.0
        c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        mu_eff = pop_size // 2
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            w = np.zeros(pop_size)
            w[:mu_eff] = np.log(pop_size / 2) - np.log(np.arange(1, mu_eff + 1))
            w_sum = np.sum(w)
            w = w / w_sum
            
            # Sample population
            population = []
            for i in range(pop_size):
                # Generate random direction
                z = np.random.randn(n)
                # Transform using C^{-1/2} approximation (use C_inv for simplicity in early iterations)
                # For proper CMA-ES we'd need eigendecomposition, but let's use a simpler approach
                direction = np.linalg.solve(C, z) if i == 0 else np.random.randn(n)
                y = x + sigma * direction
                y = np.clip(y, -5, 5)
                population.append(y)
            
            # Evaluate population
            fitnesses = []
            for y in population:
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                fy = safe_call(y)
                calls += 1
                fitnesses.append(fy)
            
            if len(fitnesses) < pop_size:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(fitnesses)
            sorted_fitnesses = [fitnesses[i] for i in sorted_idx]
            sorted_population = [population[i] for i in sorted_idx]
            
            # Update best
            if sorted_fitnesses[0] < best_f:
                best_f = sorted_fitnesses[0]
                best_x = sorted_population[0].copy()
            
            # Update mean
            new_x = np.zeros(n)
            for i in range(mu_eff):
                new_x += w[i] * sorted_population[i]
            x = new_x
            
            # Update step size
            y_sigma = (x - population[0]) / sigma if sigma > 0 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * y_sigma
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(0.001, min(10.0, sigma))
            
            # Update covariance
            if np.linalg.norm(p_sigma) / chi_n < 1.4 + 2 / (n + 1):
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * y_sigma
            else:
                p_c = (1 - c_c) * p_c
            
            delta_hs = (1 + 2 * max(0, (1.4 + 2 / (n + 1)) - np.linalg.norm(p_sigma) / chi_n)) * c_c * (2 - c_c) - 1
            rank_mu = sum(w[i] * np.outer(sorted_population[i] - x, sorted_population[i] - x) for i in range(mu_eff))
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * (np.outer(p_c, p_c) + delta_hs) + c_mu * rank_mu
            
            # Regularize C to maintain positive definiteness
            C = (C + C.T) / 2
            C += 1e-8 * np.eye(n)
            
            # Invert C
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
