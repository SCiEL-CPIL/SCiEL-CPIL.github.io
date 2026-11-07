import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    This strategy is derivative-free, which is crucial for the black-box setting.
    It adapts the shape of the search distribution to the problem's local geometry,
    making it highly effective for Rosenbrock-like functions with narrow valleys.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # 5s safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # Track best solution found so far
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                # If budget is exhausted, return a large value to discourage further selection
                # without actually calling func (which would violate the budget constraint)
                return 1e100
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        # For dim=5, log(5)~1.6, 3*1.6=4.8, floor=4. lambda=8.
        # Let's use a slightly larger population for better exploration in 5D.
        # lambda = 10 is a good balance for dim=5.
        lam = 10
        mu = lam // 2  # Number of recombination vectors
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # Standard formulas from Hansen's CMA-ES papers
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial parameters
        # Start at the center of the box, which is a reasonable guess for shifted/rotated Rosenbrock
        # though random restarts might be better if the center is far.
        # However, CMA-ES is robust to initial mean.
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size. Box is [-5,5], so 2.0 covers a good portion.
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)
        
        # Evolution path for covariance update
        p_c = np.zeros(dim)
        # Evolution path for sigma update
        p_sigma = np.zeros(dim)
        
        # Expected norm of N(0, I)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Main loop
        # We run generations until budget is exhausted or time runs out.
        # Each generation costs 'lam' function evaluations.
        
        while calls_made + lam <= budget and time.time() < deadline:
            # 1. Sample 'lam' points from the multivariate normal N(mean, sigma^2 * C)
            # We need to generate random vectors.
            # To be reproducible and fast, we can use a fixed seed or just np.random.
            # Using a local RNG is safer to avoid global state issues.
            rng = np.random.default_rng()
            
            # Generate standard normal random vectors
            zs = rng.standard_normal((lam, dim))
            
            # Transform to search distribution:
            # x_k = mean + sigma * C_inv_sqrt @ z_k
            # Since C_inv_sqrt is maintained, we can just multiply.
            # Note: C_inv_sqrt is such that C_inv_sqrt @ C_inv_sqrt.T = C^{-1}
            # Actually, standard CMA-ES uses B * D * z where C = B D^2 B^T.
            # My C_inv_sqrt variable above is effectively D^{-1} B^{-1} if C = B D^2 B^T.
            # Let's stick to the standard formulation:
            # We maintain C and its Cholesky factor or eigen decomposition.
            # For simplicity and robustness, we will update C via eigen decomposition every step.
            
            # Optimization: Instead of full Cholesky every step, we can use the
            # standard rank-1 and rank-mu updates on C directly.
            
            # Generate candidates
            # We need C^{1/2}. Let's compute it via eigen decomposition once per generation
            # or maintain it. For dim=5, eigen decomposition is very cheap.
            
            try:
                # Eigen decomposition of C
                # C is symmetric positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-12)
                # C^{1/2} = V * diag(sqrt(eigvals)) * V.T
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            except Exception:
                # Fallback if numerical issues arise
                C_sqrt = np.eye(dim)

            candidates = np.array([mean + sigma * (C_sqrt @ z) for z in zs])
            
            # Evaluate candidates
            fitnesses = np.array([tracked_func(x) for x in candidates])
            
            # 2. Sort by fitness (best first)
            sorted_idx = np.argsort(fitnesses)
            zs_sorted = zs[sorted_idx]
            fitnesses_sorted = fitnesses[sorted_idx]
            
            # 3. Update mean (weighted recombination)
            # y_i = (x_i - mean) / sigma
            y = (candidates[sorted_idx] - mean) / sigma
            
            mean_new = mean + sigma * np.sum(weights[:, None] * y[:mu], axis=0)
            
            # 4. Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt^{-1} * y_w
            # y_w is the weighted sum of y
            
            # C_sqrt_inv is needed.
            # C^{-1/2} = V * diag(1/sqrt(eigvals)) * V.T
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            
            y_w = np.sum(weights[:, None] * y[:mu], axis=0)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
            
            # Heuristic for sigma update
            # |p_sigma| / chi_n
            if np.linalg.norm(p_sigma) / chi_n < (1 + 2 * (c_sigma / (1 - c_sigma))) * c_c:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # p_c update
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * C^{-1} * y_w
            # h_sigma is a threshold to avoid covariance collapse
            
            # h_sigma = (1 + 2 * (c_sigma / (1 - c_sigma))) * c_c
            # Condition: ||p_sigma|| / chi_n < (1 + 2 * (c_sigma / (1 - c_sigma))) * c_c
            # This is the same condition as above.
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / chi_n < (1 + 2 * (c_sigma / (1 - c_sigma))) * c_c) else 0.0
            
            # C_inv is needed for p_c update
            C_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ y_w)
            
            # 5. Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma^2) * c_c * (2 - c_c) * C) + c_mu * sum(weights * y_i * y_i.T)
            
            # Term 1: Rank-1 update from p_c
            term_1 = p_c * p_c.T
            # Correction term for h_sigma
            corr = (1 - h_sigma**2) * c_c * (2 - c_c) * C
            
            # Term 2: Rank-mu update
            # sum of weighted outer products of y[:mu]
            # y_mu = y[:mu] * sqrt(weights)
            y_mu = y[:mu] * np.sqrt(weights[:, None])
            term_2 = (y_mu.T @ y_mu) * c_mu
            
            # Combine
            C = (1 - c_1 - c_mu) * C + c_1 * (term_1 + corr) + term_2
            
            # Ensure C is symmetric (numerical stability)
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Check for convergence or stagnation
            # If sigma becomes too small, we might be stuck.
            # But CMA-ES usually handles this.
            
            # If we have used a lot of budget, we might want to restart if progress is slow?
            # For now, let it run.
            
            # Safety check: if sigma is extremely small, break to avoid wasting calls on noise
            if sigma < 1e-8:
                break

        # Final Fallback: If we have remaining budget and haven't found a good solution,
        # do a simple random search or local refinement.
        # But CMA-ES should have used the budget efficiently.
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(best_x)
                calls_made += 1
        
        return best_x, best_f
