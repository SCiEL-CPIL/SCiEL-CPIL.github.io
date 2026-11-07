"""Optimized program using CMA-ES for the Bent Cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(d))
        # Initial step size sigma
        sigma = 2.0
        # Initial mean (center of the box)
        mean = np.zeros(d)
        # Covariance matrix C
        C = np.eye(d)
        # Inverse square root of C (for efficient sampling)
        # We can update this periodically or use a simple Cholesky-like update
        # For simplicity and robustness in this budget, we'll use a simplified update 
        # or rely on the standard CMA-ES update rules.
        
        # CMA-ES hyperparameters
        mu = lam // 2  # Number of parents
        # Weights (logarithmic)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        best_x = mean.copy()
        best_f = np.inf
        
        # Pre-allocate memory for population
        X = np.zeros((lam, d))
        F = np.zeros(lam)
        
        # Initial evaluation
        # We start with a small random perturbation to ensure we don't start exactly at a symmetric point if the shift is weird, 
        # but mean=0 is usually safe for [-5,5] box.
        
        calls = 0
        generation = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # 1. Sample population
            # Update inverse sqrt C approximation if needed, or just use C for sampling 
            # For small d, direct sampling is fine.
            # y ~ N(0, I)
            Y = np.random.randn(lam, d)
            
            # x = mean + sigma * C^{1/2} y
            # We need C^{1/2}. We can maintain an inverse square root matrix B such that C = B B^T.
            # Updating B is complex. Alternatively, we can just solve linear systems or use eigen decomposition periodically.
            # For d=5, eigen decomposition is cheap.
            
            # To keep it simple and fast, we will update C and use it directly.
            # Actually, standard CMA-ES updates C and uses it.
            
            # Let's use a simplified approach: 
            # We maintain C. To sample, we need sqrt(C).
            # We can compute eigendecomposition of C every generation. d=5 is small.
            evals, evecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            evals = np.clip(evals, 1e-12, None)
            sqrt_C = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            
            X = mean[None, :] + sigma * (Y @ sqrt_C.T)
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                
                if time.time() > deadline:
                    break
            
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # 3. Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # 4. Update mean
            mean = np.sum(w[:, None] * X[:mu], axis=0)
            
            # 5. Update step size sigma
            # ps update
            # We need C^{-1/2}
            inv_sqrt_C = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * C^{-1/2} * (x_1 - x_old) / sigma
            # x_1 is the best point, x_old is the old mean
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (inv_sqrt_C @ (X[0] - (mean - np.sum(w[:, None] * X[:mu], axis=0)))) / sigma
            # Note: The term in parenthesis is the new mean - old mean.
            # Actually, the update is: ps = (1-c_sigma)*ps + c_sigma*sqrt(mu_eff)*C^{-1/2} * (new_mean - old_mean)/sigma
            # Let's re-calculate:
            # old_mean was the mean before update.
            # The code above overwrote mean. Let's store old_mean.
            
            # Let's restart the loop logic slightly to store old_mean properly.
            
            # Since I modified the code structure, let's ensure correctness.
            # I will re-implement the core loop more carefully.
            
            pass # Placeholder, I will rewrite the whole __call__ in the final output.
            
        return best_x, best_f

    def _run_cma_es(self, func, d, budget, deadline):
        """Robust CMA-ES implementation."""
        lam = 4 + int(3 * np.log(d))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        best_x = mean.copy()
        best_f = np.inf
        calls = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            old_mean = mean.copy()
            
            # Sample
            # Compute eigen decomposition of C for sampling and updates
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-12, None)
            sqrt_C = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            inv_sqrt_C = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            Y = np.random.randn(lam, d)
            X = mean[None, :] + sigma * (Y @ sqrt_C.T)
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                if time.time() > deadline:
                    break
            
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update Mean
            mean = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Update Step Size (Sigma)
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * C^{-1/2} * (mean - old_mean) / sigma
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (inv_sqrt_C @ (mean - old_mean)) / sigma
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if (norm_ps / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (d + 1)) * chi_d else 0
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_d - 1))
            
            # Update Covariance Matrix
            # pc = (1 - c_c) * pc + h_sigma * c_c * sqrt(mu_eff) * (mean - old_mean) / sigma
            pc = (1 - c_c) * pc + h_sigma * c_c * np.sqrt(mu_eff) * (mean - old_mean) / sigma
            
            # Rank-one update
            diff = (mean - old_mean) / sigma
            C += c_1 * (1 - h_sigma) * (1 + (2 / (mu_eff - 1) - 1) * h_sigma) * np.outer(pc, pc)
            
            # Rank-mu update
            z = inv_sqrt_C @ (X[:mu] - old_mean).T / sigma
            C += c_mu * (np.sum(w[:, None, None] * (z[:, :, None] * z[:, None, :]), axis=0) - (1 - h_sigma) * C)
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-10 * np.eye(d)
            
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        
        # Run CMA-ES
        best_x, best_f = self._run_cma_es(func, d, budget, deadline)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
