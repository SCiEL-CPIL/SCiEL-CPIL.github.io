"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n + 1))
        # For n=5, lambda = 4 + floor(3 * log(6)) = 4 + floor(3 * 1.79) = 4 + 5 = 9
        # Let's use a smaller population to save budget for more generations if possible, 
        # but standard CMA-ES benefits from larger pop. 
        # With budget 500, and lambda=9, we get ~55 generations.
        # If lambda=5, we get 100 generations.
        # Katsuura is rugged. CMA-ES handles this well. Let's stick to standard-ish lambda.
        # Actually, for small n, smaller lambda is often fine. Let's use lambda = 4 + n = 9.
        # Or even lambda = 2*n = 10? No, standard is better.
        # Let's use lambda = 9.
        
        # However, we have a strict budget of 500 calls.
        # 500 / 9 = 55 generations.
        # 500 / 5 = 100 generations.
        # CMA-ES converges faster with larger populations but uses more calls per gen.
        # Given the ruggedness, let's try lambda = 9.
        
        # Actually, let's look at the "error_repair" hint. 
        # The hint asks about "operator 'error_repair'". This is likely a red herring or specific to the evolution system's internal logic, 
        # not necessarily a standard optimization operator I should implement literally. 
        # But "error repair" could imply a strategy of repairing bad solutions.
        # CMA-ES effectively "repairs" the search direction by adapting the covariance.
        
        # Let's implement a lightweight CMA-ES.
        
        # Initial mean
        # The problem states minimizer is in [-4, 4]^5.
        # Start at 0.
        x_mean = np.zeros(n)
        
        # Initial sigma
        # Standard CMA-ES starts with sigma = 0.25 * (max - min) or similar.
        # Range is 10. 0.25 * 10 = 2.5.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Parameters
        mu = lambda_ // 2  # Number of recombination weights
        # Weights: w_i = log(mu + 0.5) - log(i)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_cov = (2 + 2 * mu_eff / n) / (n + 1 + 3 * c_c)
        
        # Eigenvalue update rates
        # In standard CMA-ES, we update eigenvalues every n/mu_eff steps.
        # For simplicity and speed, we can use a simplified update or full matrix update.
        # Given n=5, full matrix update is cheap.
        
        # Precompute some constants
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # We will use a simplified CMA-ES that updates the full covariance matrix every generation
        # This is computationally trivial for n=5.
        
        # State
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Random number generator
        rng = np.random.default_rng(42)
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sample population
            # Need to generate lambda candidates
            # y_i = x_mean + sigma * B * z_i
            # B is the square root of C. For n=5, we can just use Cholesky or SVD.
            # Since C is PSD, Cholesky works.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-PSD due to numerical errors, reset or repair
                # "Error repair" strategy: reset C to identity or add small regularization
                C = np.eye(n)
                L = np.eye(n)
            
            # Generate standard normal samples
            Z = rng.standard_normal((lambda_, n))
            # Transform to search space
            P = np.array([L @ z for z in Z])
            X = x_mean + sigma * P
            
            # Evaluate population
            F = []
            for i in range(lambda_):
                # Clip to box
                x_i = np.clip(X[i], -5, 5)
                f_i = func(x_i)
                calls += 1
                F.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                
                # Check budget mid-population
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            # If we didn't evaluate all, adjust for next step? 
            # For simplicity, if we break early, we just exit.
            # But we should ensure we don't break in a way that wastes the remaining budget if we can't finish a gen.
            # Given 500 budget and lambda=9, we have 55 full gens. 
            # 55 * 9 = 495. 5 calls left. 
            # So we will finish 55 full generations.
            
            # Sort population by fitness
            # We only have `calls` evaluations this round. If we broke early, F is shorter.
            # But we check `if calls >= budget: break` after adding to F.
            # So if we exit the inner loop, we exit the outer loop.
            
            # Sort indices
            sorted_indices = np.argsort(F)
            # Take top mu
            I = sorted_indices[:mu]
            
            # Update mean
            # x_new = sum(w_i * x_{i:lambda})
            # Note: X[i] is the unclipped sample, but we evaluated clipped x_i.
            # CMA-ES typically uses the evaluated points.
            # However, the mapping from clipped x back to unclipped z is not 1:1 if clipping occurred.
            # For robustness, we use the evaluated x_i values for the mean update.
            # This is a deviation from pure CMA-ES but necessary for box constraints.
            
            x_new = np.zeros(n)
            for i, idx in enumerate(I):
                # We need the actual evaluated x. 
                # We stored X (unclipped) and computed x_i = clip(X[i]).
                # We should use the clipped x_i for the mean update to stay within bounds.
                x_i_eval = np.clip(X[idx], -5, 5)
                x_new += weights[i] * x_i_eval
                
            # Update sigma
            # y_sigma = (1/sigma) * (x_new - x_mean)
            y_sigma = (x_new - x_mean) / sigma
            
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * y_sigma
            
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(gen+1)}) < (1.4 + 1/n) * chi_n
            # We don't track gen count explicitly for the power term in this simplified version,
            # or we can approximate. Let's use a fixed threshold for h_sigma to simplify.
            # h_sigma = 1 if ||p_sigma|| < (1.4 + 1/n) * chi_n * sqrt(1 - (1-c_sigma)^{2*gen})
            # For simplicity, let's just use:
            # h_sigma = 1 if ||p_sigma|| / chi_n < 1.4 + 1/n
            # This is an approximation.
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1.0/n) else 0.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            c_c_sqrt = np.sqrt(c_c * (2 - c_c) * mu_eff)
            p_c = (1 - c_c) * p_c + h_sigma * c_c_sqrt * (x_new - x_mean) / sigma
            
            # Update Covariance Matrix C
            # C = (1 - c_cov * (1 + (1 - h_sigma) * mu_eff)) * C 
            #     + c_cov * (h_sigma * p_c * p_c^T + sum(w_i * y_i * y_i^T))
            # where y_i = (x_{i:lambda} - x_mean) / sigma
            
            # Calculate y_i for top mu
            Y = np.zeros((mu, n))
            for i, idx in enumerate(I):
                x_i_eval = np.clip(X[idx], -5, 5)
                Y[i] = (x_i_eval - x_mean) / sigma
            
            # Rank-one update
            rank_one = np.outer(p_c, p_c)
            
            # Rank-mu update
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(Y[i], Y[i])
            
            # Combine
            C = (1 - c_cov * (1 + (1 - h_sigma) * mu_eff)) * C + c_cov * (h_sigma * rank_one + rank_mu)
            
            # Ensure C is symmetric and PSD
            C = (C + C.T) / 2
            # Add small regularization to ensure PSD if needed
            # Check eigenvalues
            eigs = np.linalg.eigvalsh(C)
            if np.min(eigs) < 1e-10:
                C += 1e-10 * np.eye(n)
            
            # Update mean
            x_mean = x_new
            
            # Optional: If sigma becomes too small or too large, adjust
            if sigma < 1e-10:
                break
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_f
# EVOLVE-BLOCK-END
