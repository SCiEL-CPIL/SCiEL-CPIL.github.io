"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 290.0  # Safety margin before 300s/330s kill
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: lambda. For small dims, 4 + floor(3 * ln(dim)) is common.
        # For dim=5, 4 + 3*1.6 = 8.8 -> 9. Let's use a slightly larger pop for better exploration.
        # However, budget is only 500.
        # If lambda=10, we can do 50 generations.
        # If lambda=5, we can do 100 generations.
        # CMA-ES usually benefits from lambda >= 4+3*ln(d).
        # Let's try lambda = 10 for a good balance.
        lam = 10
        mu = lam // 2
        
        # Initial step size
        # Box is [-5, 5]. Initial sigma should cover a good portion.
        # Standard deviation of uniform[-5,5] is 10/sqrt(12) ~ 2.88.
        # Let's start with sigma = 2.0.
        sigma = 2.0
        
        # Initial mean: center of the box
        m = np.zeros(dim)
        
        # Initial covariance: identity
        C = np.eye(dim)
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        
        best_x = None
        best_f = np.inf
        
        # We need to be careful with the budget.
        # Each generation consumes 'lam' calls.
        # Total calls = num_generations * lam + 1 (initial evaluation? No, CMA-ES doesn't eval m initially usually, or we can).
        # Let's evaluate the initial mean to have a baseline.
        
        # Check time
        def time_left():
            return time.perf_counter() < deadline
        
        if not time_left():
             return np.zeros(dim), np.inf

        # Initial evaluation of mean
        fx = func(m)
        best_x = m.copy()
        best_f = fx
        calls = 1
        
        while calls + lam <= budget and time_left():
            # Sample population
            B = rng.standard_normal((lam, dim))
            # Cholesky of C
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-positive definite, reset
                C = np.eye(dim)
                C_chol = np.eye(dim)
            
            # Transform
            # y = B @ C_chol
            Y = B @ C_chol
            # x = m + sigma * Y
            X = m + sigma * Y
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
            
            # Sort population
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Weighted mean of best mu points
            w_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update mean
            m = m + sigma * w_mean
            
            # Update sigma (CSA)
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * C_chol.T @ (w_mean / sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_chol.T @ (w_mean / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls + 1) / lam))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            sigma *= np.exp(cs / sigma * (np.linalg.norm(ps) / np.sqrt((dim + 1) / (dim - 1)) - 1))
            
            # Update covariance (CS)
            pc = (1 - cs) * pc + hsig * cs * (2 - cs) * (w_mean * w_mean.T - C)
            
            # Update covariance (CM)
            rank_mu = np.sum(weights[:, None] * (Y_sorted[:mu] * Y_sorted[:mu].T), axis=0)
            C = (1 + c1 * hsig * (2 - cs) - c1 - cmu) * C + c1 * (pc + pc.T - pc @ pc) + cmu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Check convergence
            if np.linalg.norm(sigma * np.diag(C)) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
