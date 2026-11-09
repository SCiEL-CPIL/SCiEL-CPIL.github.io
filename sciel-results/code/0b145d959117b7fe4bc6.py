"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Multi-start CMA-ES with a 'repair' mechanism.
    
    The core idea differs from the current program (which uses a coarse grid then single CMA-ES)
    by using multiple independent CMA-ES instances from the start.
    
    The 'error_repair' mechanism (targeting the open question) is implemented as follows:
    When an offspring x' falls outside the domain [-5, 5], instead of just clipping it (which
    might map it to a flat boundary where gradients are zero or misleading), we 'repair' it
    by projecting it back into the domain in a way that preserves the direction of the step
    as much as possible, but also adds a small random jitter if it was far outside. 
    However, a more standard 'repair' in this context for black-box is often just clipping.
    
    Let's look at the specific 'error_repair' concept. In evolutionary algorithms, if a mutation
    produces an invalid individual (outside bounds), repair is needed.
    Standard repair: clip to bounds.
    Alternative repair: reflect back (like in some GA implementations).
    
    The prompt asks if 'error_repair' changes the chance of improvement.
    The current program clips. Let's try a strategy that uses **Differential Evolution (DE)**
    or a **Population-based** approach that is robust to multimodality, specifically using
    multiple populations (islands) to ensure we don't get stuck in local minima of Schwefel.
    
    Actually, the most robust strategy for BBOB Schwefel with 500 calls in 5D is usually
    a well-tuned CMA-ES. The current program's weakness is the initial grid which wastes
    budget and the single CMA-ES which might get stuck.
    
    New Strategy: **Multi-start CMA-ES with Adaptive Budget Allocation**.
    We will run 3 independent CMA-ES instances.
    Each gets a fraction of the budget.
    Crucially, we will implement a specific 'repair' logic for out-of-bounds points.
    Instead of simple clipping, if a point is out of bounds, we will evaluate the clipped version
    but also keep track of the "raw" distance. If the raw point was significantly out of bounds,
    we might penalize the fitness slightly or just rely on the fact that CMA-ES will learn to
    stay inside.
    
    Wait, the task says: "func(x) clips a finite length-5 vector to the box".
    So the function ITSELF clips. We don't need to repair before calling func.
    The 'error_repair' idea might refer to repairing the *state* of the optimizer if it drifts
    too far, or repairing the covariance matrix if it becomes ill-conditioned.
    
    Let's implement a robust CMA-ES with:
    1. Multiple starts (3 instances).
    2. Careful handling of the covariance matrix update to avoid numerical instability (repairing eigenvalues).
    3. No initial grid waste; start with reasonable sigma.
    
    This is materially different from the current "Grid + Single CMA-ES" by using "Multi-start CMA-ES with Numerical Repair".
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        
        # We will split the budget among 3 independent CMA-ES instances.
        # This increases the chance of finding the global minimum in a multimodal landscape.
        n_islands = 3
        budget_per_island = budget // n_islands
        
        best_x_overall = None
        best_f_overall = float('inf')
        
        # Seed for reproducibility but different for each island
        base_seed = 42
        
        for i in range(n_islands):
            if time.time() > deadline:
                break
                
            # Each island gets a portion of the remaining budget
            # We don't strictly need to split, we can run them sequentially
            # or interleave. Sequential is simpler and avoids context switching overhead.
            # However, interleaving might allow early termination of bad islands?
            # With only 500 calls, overhead is low. Let's run sequentially for simplicity 
            # but allocate budget dynamically? 
            # No, simple equal split is safer for this budget.
            
            island_budget = budget_per_island
            
            # Initialize CMA-ES parameters
            # For dim=5, lambda=8 or 12 is good. Let's use 10 for balance.
            lambda_ = 10
            mu = lambda_ // 2
            
            # Initial center: Random in [-4, 4]
            rng = np.random.default_rng(base_seed + i)
            x = rng.uniform(-4, 4, dim)
            
            # Initial sigma: 
            # The domain is 10 units wide. A sigma of 1.0 is a good start.
            sigma = 1.0
            
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2)**2 + mu_eff))
            
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            calls_made = 0
            best_x_island = x.copy()
            best_f_island = float('inf')
            
            # Evaluate initial point to have a baseline? 
            # CMA-ES usually starts with a population. We can evaluate the mean first.
            f_init = func(x)
            calls_made += 1
            best_f_island = f_init
            
            while calls_made < island_budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                Z = rng.standard_normal((lambda_, dim))
                Y = (C_inv_sqrt @ Z.T).T
                X = np.array([x + sigma * y for y in Y])
                
                # Clip to domain (func also clips, but good to do here for consistency)
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(xi) for xi in X])
                calls_made += lambda_
                
                # Sort
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update best
                if F[0] < best_f_island:
                    best_f_island = F[0]
                    best_x_island = X[0].copy()
                
                # Update best overall
                if best_f_island < best_f_overall:
                    best_f_overall = best_f_island
                    best_x_overall = best_x_island.copy()
                
                # Weighted mean
                m_new = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
                
                # Update sigma
                diff = m_new - x
                diff_y = C_inv_sqrt.T @ diff
                
                # h_sigma condition (simplified)
                if np.linalg.norm(p_sigma) < 1.4 * chi_n:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
                    
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(mu_eff) * c_sigma * diff_y
                
                # Sigma update
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1.0))
                
                # Update p_c
                if np.linalg.norm(p_c) < 1.4 * chi_n:
                    h_c = 1.0
                else:
                    h_c = 0.0
                    
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(mu_eff) * c_c * diff_y
                
                # Covariance update
                Diffs = X - x
                Y = (C_inv_sqrt.T @ Diffs.T).T
                Y_mu = Y[:mu]
                
                C_mu = Y_mu.T @ (weights * Y_mu)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (1 - c_c) * C) + c_mu * C_mu
                
                # Symmetrize and Repair
                C = (C + C.T) / 2
                
                # Numerical Repair: Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                # If any eigenvalue is negative or too small, clip it
                min_eig = np.min(eigvals)
                if min_eig < 1e-12:
                    # Repair by shifting eigenvalues
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs * eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                
                # Update mean
                x = m_new
                
        # If no calls were made (should not happen), return zero
        if best_x_overall is None:
            best_x_overall = np.zeros(dim)
            best_f_overall = func(best_x_overall)
            
        return best_x_overall, best_f_overall

# EVOLVE-BLOCK-END
