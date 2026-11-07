"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES Optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda = 4 + floor(3 * log(dim))
        # For dim=5: 4 + floor(3*1.609) = 4 + 4 = 8? 
        # Standard formula: lambda = 4 + floor(3 * log(n))
        # log(5) ~ 1.609, 3*1.609 = 4.827, floor = 4. lambda = 8.
        # Let's use lambda = 10 for better exploration in 5D, or stick to standard.
        # Standard CMA-ES often uses lambda = 4 + floor(3*log(n)).
        # Let's calculate: n=5, log(5)=1.609, 3*1.609=4.827, floor=4. lambda=8.
        # With budget 500, 500/8 = 62.5 generations.
        # Let's try lambda = 12 for slightly faster convergence if landscape is smooth, 
        # but 8 is safer for ruggedness. Let's stick to 10 as a middle ground.
        
        lambda_ = 10
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0  # Initial step size, covers a good part of [-5, 5]
        
        # CMA-ES internal state
        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Algorithm parameters
        mu = lambda_ // 2  # Number of parents
        # Weights: log-linear, normalized to sum 1
        w_raw = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w_raw / w_raw.sum()
        
        mu_eff = 1.0 / np.sum(w**2)
        
        # Update rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Damping for sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Expected norm of N(0, C)
        # Approximation for expected norm of N(0, I)
        # E[||N(0, I)||] = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)
        # For d=5: sqrt(2) * Gamma(3) / Gamma(2.5)
        # Gamma(3) = 2
        # Gamma(2.5) = 1.5 * 0.5 * sqrt(pi) = 0.75 * 1.772 = 1.329
        # E[||.||] = 1.414 * 2 / 1.329 = 2.127
        # Let's compute it properly using scipy if available, or just use a constant.
        # Since we only have numpy, we can use a lookup or approximation.
        # For dim=5, chi distribution mean is approx 2.127.
        chi_mean = 2.127 
        
        calls = 0
        best_x = x0.copy()
        best_f = np.inf
        
        while calls < budget:
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Sample normals
            N = np.random.randn(lambda_, dim)
            # Transform to N(x, sigma^2 C)
            Y = x + sigma * (N @ L.T)
            
            # Evaluate population
            F = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if calls >= budget:
                    break
                xi = Y[i]
                # Clip to bounds
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                F[i] = fi
                calls += 1
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            if calls >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            x_new = np.sum(w[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update covariance matrix
            # pc and ps updates
            pc = (1 - cc) * pc + cc * (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new - x) / sigma
            
            # Update C
            C = (1 + c1 + cmu - c1 - cmu) * C  # This is 1*C, so just C
            # Actually, the update is:
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cc)^2) * delta_sigma * C) + cmu * sum(w_i * (y_i - x) (y_i - x).T)
            
            # Let's do it step by step
            # First, the decay part
            C = (1 - c1 - cmu) * C
            
            # Rank-one update from pc
            C += c1 * (pc * pc.T + (1 - (1 - cc)**2) * delta_sigma * C)
            
            # Rank-mu update from parents
            # Y[:mu] are the parents relative to old x? No, we need y_i - x (old mean)
            # Wait, the standard update uses y_i - x (old mean)
            # We have Y which are the samples. We need to subtract the old x.
            Y_parents = Y[:mu] - x[np.newaxis, :]
            # Weighted sum of outer products
            # w is shape (mu,), Y_parents is (mu, dim)
            # We need sum(w_i * y_i * y_i.T)
            # This can be done as (Y_parents.T * w) @ Y_parents
            # Note: w is normalized to sum 1, but for rank-mu update, usually we use w_i directly.
            # The formula is: cmu * sum_{i=1}^mu w_i (y_i - x)(y_i - x)^T
            C += cmu * (Y_parents.T * w) @ Y_parents
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(cs / d_sigma * (norm_ps / chi_mean - 1))
            
            # Update mean
            x = x_new
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            # Check eigenvalues, if any <= 0, reset C
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals <= 0):
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sigma = 2.0 # Reset sigma? Or keep? Usually reset helps.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
