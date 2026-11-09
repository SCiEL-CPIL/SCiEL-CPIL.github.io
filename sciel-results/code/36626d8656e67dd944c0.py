"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start CMA-ES for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # Use a fixed seed for reproducibility, but vary it slightly per instance if needed? 
        # The task says "For each instance in [1, 2, 3] the evaluator creates Optimizer...".
        # So we can use a fixed seed or derive it. Fixed seed is safer for consistency.
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Multi-start strategy:
        # We split the budget into K restarts.
        # For Schwefel in 5D, multimodality is significant.
        # Let's try K=5 restarts.
        # Each restart gets budget/K calls.
        # CMA-ES population size lambda = 4 + floor(3 * log(dim)) = 4 + 3*2 = 10 for dim=5?
        # Standard lambda for CMA-ES is often 4 + 3*ln(mu) where mu = lambda/2.
        # Let's use a smaller population to fit within budget.
        # If budget per restart is 100, and lambda=10, we get 10 generations.
        # That might be too few.
        
        # Alternative: One large CMA-ES with restarts triggered by stagnation?
        # Or simply 5 independent CMA-ES instances running in parallel (sequentially) with different initializations.
        
        # Let's try 5 independent CMA-ES instances.
        # Total budget 500.
        # 5 instances * 100 calls each.
        # For dim=5, lambda=4 is small but fast. lambda=10 is better.
        # If lambda=10, 100 calls = 10 generations.
        # CMA-ES usually needs more generations.
        
        # Maybe fewer restarts? K=3?
        # 3 * 166 calls. lambda=10 -> 16 generations.
        
        # Let's try K=3 restarts with CMA-ES.
        K = 3
        calls_per_restart = budget // K
        # Reserve some calls for final polishing? No, just run them.
        
        for k in range(K):
            # Initialize center
            # Use different starting points for each restart
            # Sobol or random? Random with different seeds.
            rng_k = np.random.default_rng(42 + k)
            x = rng_k.uniform(-5, 5, dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # CMA-ES parameters
            sigma = 2.0  # Initial step size
            # Population size
            lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
            mu = lam // 2  # Number of parents
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)
            
            # Covariance matrix
            C = np.eye(dim)
            # Step size control
            cs = (mueff + 1) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
            # Damping for sigma
            dsigma = 0.05
            # Accumulation path
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            calls_used = 1  # Initial point
            # We have calls_per_restart - 1 calls left for this instance
            
            while calls_used < calls_per_restart and time.time() < deadline:
                # Generate population
                B = rng_k.multivariate_normal(np.zeros(dim), C, size=lam)
                # Update paths
                # Sort by fitness? No, we need to evaluate first.
                # Evaluate population
                offspring = np.clip(x + sigma * B, -5, 5)
                offspring_f = np.array([func(xi) for xi in offspring])
                calls_used += lam
                
                # Update best
                idx_min = np.argmin(offspring_f)
                if offspring_f[idx_min] < best_f:
                    best_f = offspring_f[idx_min]
                    best_x = offspring[idx_min].copy()
                
                # Sort offspring by fitness
                order = np.argsort(offspring_f)
                y = offspring[order]
                f = offspring_f[order]
                
                # Mean update
                y_w = np.sum(w[:, None] * y[:mu], axis=0)
                x_old = x
                x = np.clip(x + y_w, -5, 5)
                # Note: Clipping the mean might break CMA-ES assumptions slightly, 
                # but it's necessary to stay in bounds.
                
                # Update sigma (step size adaptation)
                # Calculate norm of y_w / sigma
                norm_yw = np.linalg.norm(y_w)
                if norm_yw > 1e-12:
                    ps = (1 - cs) * ps + cs / np.sqrt(1 - (1 - cs)**2) * (y_w / sigma)
                else:
                    ps = (1 - cs) * ps
                
                # Update covariance
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chiN) < (1 + 2/(dim+1)) * (1 - cs)
                pc = (1 - c1) * pc + hsig * c1 * (1 + c1 - cmu) * np.outer(ps, ps)
                
                # Update C
                delta_d = (1 - c1) * (1 + c1 - cmu) + hsig * c1 * cmu * (np.trace(pc) - dim)
                C = (1 - c1 - cmu) * C + c1 * delta_d * pc + cmu * np.sum(w[:, None, None] * np.outer(y[:mu], y[:mu]) / sigma**2, axis=0)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Re-orthogonalize if needed? CMA-ES usually handles this.
                # Add small regularization to prevent singularity
                C += 1e-9 * np.eye(dim)
                
                # Update sigma
                sigma *= np.exp(csigma * (np.linalg.norm(ps) / chiN - 1))
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
