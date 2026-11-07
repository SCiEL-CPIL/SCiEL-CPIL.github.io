import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard lambda for small d
        lam = 4 + int(3 * np.log(d))
        # Weights: mu = lam/2
        mu = lam // 2
        # Weights w_i
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial state
        # Start at the center of the box, which is a good prior for BBOB
        x_mean = np.zeros(d)
        # Initial sigma: The box is [-5, 5], width 10. 
        # A sigma of 1.0 allows coverage of [-2, 2] with high prob, 
        # but we might need to expand. Let's start with 1.5 to be safe 
        # given the optimum is in [-4, 4].
        sigma = 1.5
        C = np.eye(d)
        C_sqrt_inv = np.eye(d) # Inverse square root of C (approx identity initially)
        ps = np.zeros(d) # Step size control
        pc = np.zeros(d) # Path for covariance adaptation
        
        # Pre-compute constants
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        evals = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin for 300s limit
        
        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate lambda samples
            # We use a simple Cholesky decomposition for C_sqrt
            # For d=5, this is very fast.
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite (numerical issues), reset
                C = np.eye(d)
                C_chol = np.eye(d)
                
            # Standard normal samples
            Z = np.random.randn(lam, d)
            # Transform to x_samples
            # x_i = x_mean + sigma * (C_chol @ Z_i^T)^T
            # Note: Z is (lam, d), C_chol is (d, d)
            # We want each row of Z transformed by C_chol
            X = x_mean + sigma * (Z @ C_chol.T)
            
            # Evaluate fitness
            F = np.array([func(x) for x in X])
            evals += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
                
            # Check if we used all budget
            if evals >= budget:
                break
                
            # Sort samples by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Recombination
            y_weighted = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update mean
            x_mean = y_weighted
            
            # Update step size (sigma)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt_inv @ y_weighted
            # We need C_sqrt_inv @ y_weighted. 
            # Since C = C_chol @ C_chol.T, C_sqrt_inv = inv(C_chol.T) @ inv(C_chol)
            # Actually, it's easier to compute:
            # v = C_sqrt_inv @ (y_weighted / sigma)
            # For d=5, we can just invert C_chol.T
            try:
                C_sqrt_inv_curr = np.linalg.inv(C_chol.T)
            except np.linalg.LinAlgError:
                C_sqrt_inv_curr = np.eye(d)
                
            y_scaled = y_weighted / sigma
            v = C_sqrt_inv_curr @ y_scaled
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * v
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # h_sigma = (||ps|| / (1 - (1 - c_sigma)^(1/(evals_gen+1)))) / (sqrt(1 - (1 - c_sigma)^(2*(evals_gen+1))) * chi_d)
            # Simplified h_sigma check for small d and few generations
            # We can approximate h_sigma = 1 if ||ps|| < (1.4 + 1/d) * chi_d
            h_sigma = 1.0 if np.linalg.norm(ps) < (1.4 + 1/d) * chi_d else 0.0
            
            # Delta H term
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Update C
            # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * I) 
            #    + c_mu * sum w_i (y_i/sigma @ y_i/sigma.T)
            
            # We need y_i / sigma for the top mu individuals
            Y_scaled = X_sorted[:mu] / sigma
            # Outer products
            # Sum w_i * (y_i @ y_i.T)
            # w_i is shape (mu,), Y_scaled is (mu, d)
            # Weights expanded
            W = weights[:mu]
            # Sum of outer products
            # np.einsum or loop
            outer_sum = np.zeros((d, d))
            for i in range(mu):
                outer_sum += W[i] * np.outer(Y_scaled[i], Y_scaled[i])
                
            C = (1 - c_1 - c_mu + delta_h) * C \
                + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(d)) \
                + c_mu * outer_sum
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0

        return best_x, best_f
