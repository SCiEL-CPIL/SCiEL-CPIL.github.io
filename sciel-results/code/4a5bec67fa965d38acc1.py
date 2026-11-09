"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by CMA-ES population-based refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: finish well before 330s limit (evaluator kills at 330s)
        # Use 300s as internal deadline to be safe.
        deadline = start_time + 300.0
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility of the search strategy,
        # but the landscape is seeded by the evaluator.
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Global Scan
        # Stratified sampling to cover [-5, 5]^5
        n_scan = min(50, budget // 4)  # Use ~25% of budget for exploration

        # Generate stratified samples using Latin Hypercube approach
        n_samples = n_scan
        # Divide each dimension into bins and sample one point per bin
        # For 50 samples in 5D, we can use 5 samples per dimension
        n_per_dim = int(np.ceil(n_samples ** (1.0 / dim)))
        
        # Create Latin Hypercube
        scan_points = []
        for i in range(n_samples):
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling: divide [0,1] into n_per_dim bins
                # Pick a random position within the bin
                bin_idx = (i + d * n_samples // n_per_dim) % n_per_dim
                frac = (bin_idx + rng.uniform()) / n_per_dim
                x[d] = frac * 10.0 - 5.0
            scan_points.append(x)

        best_fx = float('inf')
        best_x = None
        candidates = []

        for x in scan_points:
            if not time_ok():
                break
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not candidates:
            # Fallback if we ran out of time immediately
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            return x0, fx0

        # Sort candidates to get the best starting points for local search
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES refinement
        remaining_budget = budget - len(candidates)

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lambda_pop = 4 + int(3 * np.log(dim))
        lambda_pop = max(lambda_pop, 10)  # At least 10 individuals
        
        # Initial sigma: based on the spread of good candidates
        # Use the distance between best and 2nd best as a heuristic
        if len(candidates) >= 2:
            init_sigma = np.linalg.norm(candidates[0][1] - candidates[1][1])
            init_sigma = max(0.1, min(init_sigma, 2.0))
        else:
            init_sigma = 1.0

        # CMA-ES state
        mean = candidates[0][1].copy()
        C = np.eye(dim)
        sigma = init_sigma
        alpha_mu = 0.5  # Learning rate for mean
        alpha_sigma = 0.5  # Learning rate for sigma
        
        # Weighting for recombination
        mu = lambda_pop // 2  # Number of best individuals used for recombination
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky factor of C
        C_chol = np.linalg.cholesky(C)
        C_chol_inv = np.linalg.inv(C_chol)

        eval_count = 0
        no_improvement = 0
        stall_threshold = 5 * lambda_pop  # Restart if no improvement for this many evals

        while time_ok() and remaining_budget > lambda_pop:
            # Generate population
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = mean + sigma * C_chol @ z
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)

            # Evaluate offspring
            offspring_fx = []
            for y in offspring:
                if not time_ok() or remaining_budget <= 0:
                    break
                fy = func(y)
                offspring_fx.append((fy, y))
                remaining_budget -= 1
                eval_count += 1

            if not offspring_fx:
                break

            # Sort offspring
            offspring_fx.sort(key=lambda t: t[0])
            
            # Check for improvement
            best_offspring_fx = offspring_fx[0][0]
            if best_offspring_fx < best_fx - 1e-10:
                best_fx = best_offspring_fx
                best_x = offspring_fx[0][1].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Update mean using weighted recombination
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(offspring_fx))):
                new_mean += weights[i] * offspring_fx[i][1]
            
            # Update path variables
            C_sqrt_inv = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
            C_sqrt_inv = np.linalg.inv(C_sqrt_inv)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Update sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (eval_count + 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - mu_eff) else 0.0
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            delta_sigma = h_sigma * (1 / c_sigma - 1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(min(mu, len(offspring_fx))):
                diff = offspring_fx[i][1] - mean
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c_1 * (delta_sigma - 1) + c_mu) * C - c_1 * rank_one - c_mu * rank_mu
            C = 0.5 * (C + C.T)  # Symmetrize

            # Update Cholesky factor
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)

            mean = new_mean

            # Stagnation detection: if no improvement for too long, restart from best
            if no_improvement > stall_threshold:
                mean = best_x.copy()
                sigma = 0.5
                C = np.eye(dim)
                C_chol = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improvement = 0

        # Final polish: use small steps around the best solution
        if time_ok() and remaining_budget > 5:
            polish_budget = min(remaining_budget, 20)
            x = best_x.copy()
            fx = best_fx
            for _ in range(polish_budget):
                if not time_ok() or remaining_budget <= 0:
                    break
                y = x + 0.01 * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                remaining_budget -= 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
