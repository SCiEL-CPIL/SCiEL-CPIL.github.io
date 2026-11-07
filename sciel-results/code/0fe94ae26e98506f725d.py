import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        
        # CMA-ES parameters
        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        # Initialization
        x = np.zeros(n) # Start at 0, usually a good guess for BBOB
        # Alternatively: x = rng.uniform(-5, 5, n)
        # Let's use a random start to be safe against shifts, but 0 is often better for average performance if shift is small.
        # The prompt says shift is hidden, minimizer in [-4,4]. Random is safer.
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        
        # CMA-ES weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Covariance matrix adaptation
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Evaluation budget tracking
        calls = 0
        best_x = x
        best_f = float('inf')
        
        # Initial evaluation
        best_f = func(x)
        calls += 1
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            B = rng.standard_normal((lam, n))
            # Whitening matrix (inverse sqrt of C)
            # Using eigen decomposition for simplicity and stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            # Cholesky or sqrt
            # C^{-1/2} = V diag(1/sqrt(lambda)) V^T
            # For sampling: y = sigma * C^{1/2} z
            # C^{1/2} = V diag(sqrt(lambda)) V^T
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample
            Y = x + sigma * (B @ sqrt_C.T) # Note: B is (lam, n), sqrt_C is (n, n). B @ sqrt_C.T is (lam, n)
            
            # Evaluate
            F = []
            for y in Y:
                y_clipped = np.clip(y, -5, 5)
                fy = func(y_clipped)
                F.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y_clipped.copy()
                if calls >= self.budget or time.time() > deadline:
                    break
            
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Sort
            F_sorted_idx = np.argsort(F)
            Y_sorted = Y[F_sorted_idx]
            F_sorted = np.array(F)[F_sorted_idx]
            
            # Update mean
            x_new = np.sum(w[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            y_invC = (Y_sorted[:mu] - x) / sigma
            # ps = (1 - c_sigma) ps + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} * w^T y
            # C^{-1/2} = V diag(1/sqrt(lambda)) V^T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (w[:mu] @ y_invC))
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - h_pc) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            h_pc = (1 - np.exp(-2 + 5 / (n + 1))) if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) / d_sigma)) < (1.4 + 2 / (n + 1)) * np.trace(C) / n else 0.0
            
            # This h_pc calculation is tricky with calls. Standard CMA-ES uses generation counter.
            # Let's simplify: just use standard update without h_pc heuristic for now, or assume h_pc=1 if not converged.
            # Actually, for robustness, let's use a simpler variance reduction or just standard CMA-ES logic.
            
            # Standard update:
            delta_h = (1 - np.exp(-2 + 5 / (n + 1)))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((calls // lam + 1) / d_sigma)) < (1.4 + 2 / (n + 1)) * np.trace(C) / n:
                h_pc = delta_h
            else:
                h_pc = 0.0
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            for i in range(mu):
                y_i = y_invC[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Clip x to box? No, CMA-ES handles unbounded, but func clips.
            # We should keep x within reasonable bounds to avoid numerical issues, but func clips anyway.
            # However, if x goes far out, sigma might explode.
            # Let's clip x to [-10, 10] or similar to be safe.
            x = np.clip(x, -10, 10)
            
        return best_x, best_f
