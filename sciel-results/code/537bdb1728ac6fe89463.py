"""Rewritten optimizer: CMA-ES with multi-start basin discovery and adaptive covariance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start initialization to handle shifted/rotated BBOB instances."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def counted_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            calls_used += 1
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: Coarse Basin Discovery ---
        # For f6 (Attractive Sector), the global minimum is often in a specific sector.
        # We sample diverse points to identify the best basin before committing to CMA-ES.
        
        # 1. Origin
        counted_func(np.zeros(dim))
        
        # 2. Axis-aligned points (covers +/- sectors)
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = 3.0
            p2 = np.zeros(dim); p2[i] = -3.0
            counted_func(p1)
            counted_func(p2)
            
        # 3. Random points in the box
        # Use remaining budget for random sampling to find a good starting point for CMA-ES
        # We reserve ~80% of budget for CMA-ES, use 20% for discovery
        discovery_budget = max(10, int(budget * 0.2))
        current_discovery_calls = calls_used
        
        while calls_used < discovery_budget and time.time() < deadline:
            p = rng.uniform(-4, 4, dim)
            counted_func(p)

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: CMA-ES Refinement ---
        # Start CMA-ES from the best point found in Phase 1
        
        x = best_x.copy()
        
        # CMA-ES Parameters
        # Lambda (population size)
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Step-size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Mean adaptation
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path lengths
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial sigma: small enough to stay local, large enough to explore basin
        # Since we started from a good point, we can use a moderate sigma
        sigma = 1.0
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C
        def compute_C_inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
        C_inv_sqrt = np.eye(dim)
        
        # Main CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            pop_f = []
            
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y_clipped = np.clip(y, -5, 5)
                
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                fy = counted_func(y_clipped)
                pop.append(y_clipped)
                pop_f.append(fy)

            if len(pop) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Update best
            # (counted_func already updates best_f/best_x)
            
            # Mean adaptation
            x_old = x.copy()
            for i in range(mu):
                x = x + weights[i] * (pop[i] - x_old)

            # Compute evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma
            
            # h_sigma (indicator for covariance update)
            # Using a simplified h_sigma based on norm of p_sigma
            # Ideally uses (1 - (1 - c_sigma)^(2 * gen)) but we approximate
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for the threshold
            h_sigma = 1.0 if norm_p_sigma / chi_n < (1.4 + 2.0 / (dim + 1)) else 0.0
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = 1.0 / (sigma ** 2) * np.outer(x - x_old, x - x_old)
            
            C = (1 + c_c * delta_h - c_c - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(pop[i] - x_old, pop[i] - x_old) for i in range(mu))

            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T

            # Update inverse square root periodically to save time
            # Every 5 generations
            if (budget - calls_used) % (lambda_ * 5) == 0 or calls_used % (lambda_ * 5) == 0:
                 # This logic is a bit rough, let's just update every few steps
                 # Actually, updating every step is safer for accuracy but slower.
                 # Let's update every step since dim=5 is small.
                 C_inv_sqrt = compute_C_inv_sqrt(C)

            # Step-size adaptation
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for convergence
            if sigma < 1e-8 or np.linalg.norm(x - x_old) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
