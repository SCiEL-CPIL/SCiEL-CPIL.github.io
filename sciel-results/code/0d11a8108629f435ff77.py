"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        # Parameters
        n = self.dim
        lambda_pop = max(4 * n, 10)  # Population size
        mu = lambda_pop // 2  # Number of parents for recombination
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c1
        
        # Initial values
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        def cholesky_with_fallback(A):
            try:
                L = np.linalg.cholesky(A)
                return L
            except np.linalg.LinAlgError:
                # Add small regularization
                L = np.linalg.cholesky(A + 1e-10 * np.eye(A.shape[0]))
                return L

        # Update C and C_inv, C_sqrt, C_sqrt_inv
        def update_matrices():
            nonlocal C, C_inv, C_sqrt, C_sqrt_inv
            # Regularize C to ensure positive definiteness
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(n)
            
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.pinv(C)
            
            C_sqrt = cholesky_with_fallback(C)
            C_sqrt_inv = np.linalg.inv(C_sqrt)

        update_matrices()
        
        best_x = x.copy()
        best_f = func(x)
        budget_used = 1
        
        gen = 0
        while budget_used < self.budget and time.time() < deadline:
            # Generate offspring
            ys = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate offspring
            fs = []
            for y in ys:
                if time.time() > deadline:
                    break
                fy = func(y)
                fs.append(fy)
                budget_used += 1
            
            if len(fs) < lambda_pop:
                # Pad with worst value if we ran out of time/budget mid-generation
                # This shouldn't happen often if we check before, but just in case
                while len(fs) < lambda_pop:
                    fs.append(float('inf'))
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * ys[i]
            
            # Update path variables
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt_inv @ (x_new - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x) / sigma
            
            # Update Covariance Matrix
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((n, n))
            for i in range(mu):
                zi = C_sqrt_inv @ (ys[i] - x) / sigma
                rankmu += weights[i] * np.outer(zi, zi)
            
            C = (1 + c1 * (hsig * (1 + cc) - cc) - c1) * C + c1 * rank1 + cmu * rankmu
            
            # Update Step Size
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(cs / dsigma * (ps_norm / np.sqrt(n) - 1))
            # Clamp sigma to avoid numerical issues
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update x
            x = x_new
            update_matrices()
            gen += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
