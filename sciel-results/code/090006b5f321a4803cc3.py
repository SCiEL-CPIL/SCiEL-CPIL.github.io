"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # Global best tracking
        best_x = None
        best_f = float('inf')
        
        # Number of independent restarts
        n_restarts = 10
        
        # Budget per restart (roughly)
        # We will dynamically allocate based on time, but let's start with a fixed split
        # 500 calls / 10 restarts = 50 calls per restart.
        # However, CMA-ES needs at least (dim+1) calls to start.
        # Let's try to give each restart a chance to explore.
        
        # To maximize AOCC, we want early progress.
        # A single long run might get stuck. Multiple short runs might find the basin faster.
        # Let's use a "race" strategy: run several CMA-ES instances in parallel (sequentially in code, but logically distinct).
        
        # We will implement a lightweight CMA-ES or (mu+lambda)-ES variant for each restart.
        # Given the small dim (5) and budget, a simple (1+1)-ES with random restarts might be sufficient,
        # but the prompt suggests CMA-ES as a different family.
        
        # Let's implement a simplified CMA-ES-like strategy with adaptation.
        # Actually, for dim=5 and budget=500, a well-tuned (1+1)-ES with restarts is often competitive.
        # But let's stick to the "different strategy" directive: Multi-start CMA-ES.
        
        # Simplified CMA-ES logic:
        # - Mean m
        # - Covariance matrix C (or step size sigma for simplicity if we assume isotropic, but CMA-ES adapts C)
        # - For dim=5, full CMA-ES is feasible.
        
        # To keep it robust and fast, we'll use a simplified version:
        # We'll use a (mu+lambda)-ES with covariance matrix adaptation (CMA-ES)
        
        def cma_es_run(func, budget_run, start_x, rng):
            # Parameters
            mu = max(2, dim // 2)  # Number of parents
            lambda_ = max(4, 2 * mu)  # Number of offspring
            # Step size
            sigma = 2.0
            # Mean
            m = start_x.copy()
            # Covariance matrix (start with identity)
            C = np.eye(dim)
            # Eigen decomposition for sampling
            # We'll update C using rank-one and rank-mu updates
            
            # Standard CMA-ES parameters
            cm = 1 / (5 * dim + sigma)  # Not standard, let's use standard values
            # Standard CMA-ES:
            # c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            # c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            # c_1 = 2 / ((dim + 1.3)^2 + mu_eff)
            # c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)^2 + mu_eff))
            
            mu_eff = mu / 2
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Initial weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            
            # Path variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Pre-compute inverse sqrt of C for sampling? 
            # For dim=5, we can just do eigendecomposition each step or use Cholesky.
            # Cholesky is faster for sampling.
            
            best_local_x = m
            best_local_f = func(m)
            
            # We need to handle the budget carefully.
            # Each generation costs lambda_ calls.
            # We have budget_run calls.
            # Generations = budget_run // lambda_
            
            # If budget_run is small, we might not even do one generation.
            # Let's ensure at least one evaluation if possible.
            
            calls_made = 1 # for m
            
            while calls_made + lambda_ <= budget_run and time.time() < deadline:
                # Sample lambda_ points
                X = np.zeros((lambda_, dim))
                F = np.zeros(lambda_)
                
                # Eigen decomposition of C
                # C is symmetric positive definite
                try:
                    D, V = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-12))
                    B = V / D  # B = C^{-1/2}
                except np.linalg.LinAlgError:
                    break
                
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = m + sigma * B @ z
                    X[i] = np.clip(y, -5, 5)
                    F[i] = func(X[i])
                    calls_made += 1
                    if F[i] < best_local_f:
                        best_local_f = F[i]
                        best_local_x = X[i].copy()
                
                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                # Update mean
                m_new = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
                
                # Update covariance
                # Rank-one update
                y_sigma = (m_new - m) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_))) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_))))
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
                
                delta_h = 1 - h_sigma
                C = (1 + c_1 + c_mu - c_1) * C + c_1 * np.outer(y_sigma, y_sigma) + c_mu * np.sum(w[:, np.newaxis] * np.outer(X[:mu] - m, X[:mu] - m) / sigma**2, axis=0) - delta_h * c_c * (2 - c_c) * np.outer(p_c, p_c)
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_))) - np.sqrt(dim * (1 - 1/(dim+1)))))
                
                m = m_new
                
                # Check for stagnation or failure
                if not np.isfinite(sigma) or sigma < 1e-12:
                    break
                
            return best_local_x, best_local_f

        # Main loop: Run multiple restarts
        # We need to distribute the budget.
        # Let's give each restart an equal share of the budget, but stop if time runs out.
        # Also, we should track the global best.
        
        rng = np.random.default_rng(0)
        
        # To improve early progress, we can run the restarts in a "round-robin" fashion or just sequentially.
        # Sequential is simpler. 
        # Let's allocate budget based on remaining budget.
        
        remaining_budget = budget
        for i in range(n_restarts):
            if remaining_budget <= 0 or time.time() >= deadline:
                break
            
            # Allocate budget to this restart
            # We want to leave some budget for potential later restarts if this one fails fast?
            # Or just split evenly?
            # Let's split evenly among remaining restarts.
            budget_for_this = remaining_budget // (n_restarts - i)
            if budget_for_this < 10: # Minimum viable budget for CMA-ES
                budget_for_this = remaining_budget
                remaining_budget = 0
            else:
                remaining_budget -= budget_for_this
            
            # Start point
            # Random uniform in [-5, 5]
            start_x = rng.uniform(-5, 5, dim)
            
            # Run CMA-ES
            try:
                x_res, f_res = cma_es_run(func, budget_for_this, start_x, rng)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res
            except Exception:
                pass
        
        # If we still have budget and time, do a final (1+1)-ES polish from the best point?
        # This can help with fine-tuning.
        if best_x is not None and remaining_budget > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(remaining_budget):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if fx < best_f:
                    best_f = fx
                    best_x = x
        
        return best_x, best_f

# EVOLVE-BLOCK-END
