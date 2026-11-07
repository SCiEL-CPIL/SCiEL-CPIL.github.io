import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For n=5, a population of 10 is reasonable.
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial State
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x
        best_f = func(x)
        calls = 1
        
        # Generations
        max_gen = (budget - 1) // lam
        
        for gen in range(max_gen):
            # Eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)
            
            # Sample
            Z = np.random.randn(lam, n)
            Y = x + sigma * (Z * sqrt_eigvals) @ eigvecs.T
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lam
            
            # Update best
            if F[0] < best_f: # F is sorted? No, not yet.
                pass
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Update Mean
            x = np.sum(w[:, None] * Y[:mu], axis=0)
            
            # Update Paths
            y_mean = np.sum(w[:, None] * Y[:mu], axis=0) - x
            
            # p_sigma update
            # Simplified h_sigma = 1
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_mean / sigma)
            
            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean / sigma)
            
            # Update Sigma
            # Check for divergence? 
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate target norm
            target = np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
            # If norm_p_sigma > target * 1.4, we might have a step size increase that is too large?
            # Standard CMA uses h_sigma. Let's implement a basic check.
            # If norm_p_sigma is very large, sigma might explode.
            # For now, simple update:
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / target - 1))
            
            # Update C
            delta_sig = 1 - (1 - c_c)**2
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + delta_sig * C)
            
            for i in range(mu):
                d = Y[i] - x
                C += c_mu * w[i] * (d @ d.T)
                
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals_c, _ = np.linalg.eigh(C)
            if np.min(eigvals_c) < 1e-12:
                C += 1e-12 * np.eye(n)

        return best_x, best_f
