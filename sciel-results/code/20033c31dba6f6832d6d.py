"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        
        # Initial state
        x = rng.uniform(-5, 5, self.dim)
        fvals = [func(x)]
        best_x = x.copy()
        best_f = fvals[0]
        
        # CMA-ES parameters
        # For dim=5, lambda is typically 4 + floor(3 log dim) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(self.dim)) 
        if lam < 2: lam = 2
        
        # Initial sigma
        sigma = 1.0
        
        # Covariance matrix C = I initially
        C = np.eye(self.dim)
        
        # Learning rates
        # cmu = (mu_eff + 2) / (lambda + 4)
        # c1 = 2 / ((dim + 1.3)^2 + lam)
        # cs = (mu_eff + 2) / ((dim + lam) * 2 + 1)
        # c_sigma = cs
        
        mu_eff = lam / 2.0
        cmu = (mu_eff + 2) / (lam + 4)
        c1 = 2 / ((self.dim + 1.3)**2 + lam)
        cs = (mu_eff + 2) / ((self.dim + lam) * 2 + 1)
        c_sigma = cs
        
        # Eigenvalues of C (initially 1)
        eigval = np.ones(self.dim)
        eigvec = np.eye(self.dim)
        
        # Counters
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # Evolution path
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Weights for recombination (top mu = lam/2)
        w = np.zeros(mu_eff)
        sum_w = 0.0
        for i in range(int(mu_eff)):
            w[i] = np.log(mu_eff + 0.5) - np.log(i + 1)
        sum_w = np.sum(w)
        w = w / sum_w
        mu_eff = 1.0 / np.sum(w**2)
        
        # Recalculate rates with correct mu_eff
        cmu = (mu_eff + 2) / (lam + 4)
        c1 = 2 / ((self.dim + 1.3)**2 + lam)
        cs = (mu_eff + 2) / ((self.dim + lam) * 2 + 1)
        c_sigma = cs
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        
        calls_made = 1
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = np.zeros((lam, self.dim))
            fs = np.zeros(lam)
            
            for i in range(lam):
                # Update eigen-decomposition occasionally or use Cholesky
                # For simplicity and speed in dim=5, we can use a simpler update
                # or just use the matrix directly. 
                # Actually, standard CMA-ES updates C. Let's stick to standard.
                
                # We need to sample from N(0, C).
                # Use eigvec * diag(sqrt(eigval)) * z
                # But eigval/eigvec need updating.
                
                # To keep code concise and robust, let's use a simpler 
                # "Diagonal CMA-ES" or just standard CMA with periodic eigen-decomp.
                
                # Let's do a simplified CMA that updates C directly.
                
                z = rng.standard_normal(self.dim)
                # Sample y = eigvec * (sqrt(eigval) * z)
                y = eigvec @ (np.sqrt(eigval) * z)
                
                yi = np.clip(x + sigma * y, -5, 5)
                fi = func(yi)
                ys[i] = y
                fs[i] = fi
                calls_made += 1
                
                if fi < best_f:
                    best_f = fi
                    best_x = yi.copy()
                
                if calls_made >= self.budget:
                    break
                    
            if calls_made >= self.budget:
                break
                
            # Sort offspring
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Weighted mean
            m = np.sum(w[:, np.newaxis] * ys[:int(mu_eff)], axis=0)
            
            # Update evolution paths
            # Inverse square root of C approx
            # For dim 5, we can compute inv_sqrt_C via eigen-decomp
            # But we already have eigval, eigvec.
            # inv_sqrt_C = eigvec @ diag(1/sqrt(eigval)) @ eigvec.T
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (eigvec.T @ (eigvec @ m)) # Approx? No.
            # Correct update: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * (C^-1/2 * (x_new - x_old)/sigma)
            # Here m is the mean of y's. So y_mean = m.
            # C^-1/2 is needed.
            
            # Let's compute C^-1/2 properly using current eigval/eigvec
            inv_sqrt_C = eigvec @ (np.diag(1.0 / np.sqrt(eigval))) @ eigvec.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (inv_sqrt_C @ m)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_made - lam))) < (1 + 2 / (self.dim + 1)) * chiN)
            
            if hsig:
                pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mu_eff) * m
            else:
                pc = (1 - c1) * pc
                
            # Update C
            # Delta_h = (2 * hsig - 1) * c1 * (1 - c1)
            delta_h = (2 * hsig - 1) * c1 * (1 - c1)
            
            # Rank-one update from m
            # Rank-mu update from top mu
            rank_one = np.outer(m, m)
            
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(int(mu_eff)):
                rank_mu += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 + cmu * c1 * delta_h - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Update x
            x = x + sigma * m
            
            # Update sigma
            # sigma *= exp(cs/dlambda * (||ps||/chiN - 1))
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chiN - 1))
            
            # Update eigen-decomposition periodically or every step?
            # For dim=5, every step is cheap enough.
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.clip(eigval, 1e-10, None)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
