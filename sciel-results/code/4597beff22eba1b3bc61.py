"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Stratified lattice sampling to identify basins
        # Use a 3^5 = 243 point grid over [-4, 4]^5, but we only have 500 calls total
        # We'll use a subset: 3 points per dimension = 243 points is too many for 500 budget
        # Let's use 2^5 = 32 points first, then add some random points
        # Actually, let's use a smarter approach: 3^3 * 2^2 = 72 points (stratified in 3 dims, 2 in others)
        # Or just use 40 random points + 20 stratified points
        
        # Stratified sampling: divide [-4,4] into 2 parts per dimension = 2^5 = 32 points
        # Plus 20 random points for coverage
        num_stratified = 32
        num_random = 20
        initial_points = []
        
        # Stratified points: 2 per dimension
        for i in range(2 ** self.dim):
            if calls >= self.budget or time.time() > deadline:
                break
            coords = []
            idx = i
            for d in range(self.dim):
                coords.append(idx % 2)
                idx //= 2
            # Map to [-4, 4] with some jitter to avoid exact grid points
            x = np.array([coords[d] * 8.0 - 4.0 + rng.uniform(-0.5, 0.5) for d in range(self.dim)])
            fx = evaluate(x)
            if fx is None:
                break
            initial_points.append((fx, x.copy()))
        
        # Random points
        for i in range(num_random):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, size=self.dim)
            fx = evaluate(x)
            if fx is None:
                break
            initial_points.append((fx, x.copy()))
        
        # Sort by fitness and take top K basins
        initial_points.sort(key=lambda item: item[0])
        top_k = min(8, len(initial_points))
        
        # Phase 2: CMA-ES from top basins
        # CMA-ES parameters
        pop_size = 10  # lambda
        sigma_min = 0.01
        sigma_max = 5.0
        max_local_steps = 25  # Max generations per CMA-ES instance (reduced to allow more basins)
        
        for idx in range(top_k):
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Start from the idx-th best initial point
            f_start, x_start = initial_points[idx]
            x_mean = x_start.copy()
            sigma = 0.8
            
            # CMA-ES state
            C = np.eye(self.dim)
            C_sqrt = np.eye(self.dim)
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Adaptation parameters
            mu = pop_size // 2  # Number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((self.dim + 2) ** 2 + mu_eff))
            
            stall_count = 0
            best_in_cma = f_start
            for gen in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(self.dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    offspring.append((fy, y))
                
                if len(offspring) < pop_size:
                    break
                
                # Sort offspring by fitness
                offspring.sort(key=lambda item: item[0])
                
                # Select best mu individuals
                selected = offspring[:mu]
                
                # Track best in this CMA-ES run
                if selected[0][0] < best_in_cma:
                    best_in_cma = selected[0][0]
                
                # Check for stall
                if best_in_cma >= f_start * 0.999 and gen > 3:
                    stall_count += 1
                    if stall_count >= 5:
                        break
                else:
                    stall_count = 0
                
                # Update mean
                new_mean = sum(weights[i] * selected[i][1] for i in range(mu))
                
                # Update covariance matrix
                y_vec = (new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) /
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1 + 3 / self.dim) * np.sqrt(1 - (1 - c_sigma) ** 2)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
                
                delta_h = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) /
                          np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) < (1 + 2 / self.dim) * np.sqrt(1 - (1 - c_c) ** 2)
                
                C = ((1 + c_1 + c_mu - c_1) * C +
                    c_1 * (np.outer(p_c, p_c) - C) +
                    c_mu * sum(weights[i] * np.outer(selected[i][1] - x_mean, selected[i][1] - x_mean) for i in range(mu)) / sigma ** 2)
                
                # Ensure positive semi-definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                # Update x_mean
                x_mean = new_mean
                
                # Update C_sqrt via Cholesky decomposition (approximate)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # Fallback to eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Phase 3: Final polish with local (1+1)-ES on the best point
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.15
            for step in range(100):
                if calls >= self.budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.05, 1.0)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.95, 0.005)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
