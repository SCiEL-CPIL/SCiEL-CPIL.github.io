"""Optimized solver for BBOB f18 using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))  # Population size
        mu = lambda_ // 2                  # Number of weighted points
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        
        # Step size control parameters
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(c1 / 2, (1 - c1) / (2 * (n + 1)))
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        
        # Initialization
        rng = np.random.default_rng(0)
        m = rng.uniform(-2, 2, n)  # Start near center, smaller range
        sigma = 1.0
        C = np.eye(n)
        C_inv = np.eye(n)
        # Precompute sqrt of C for sampling
        # We need to update this every generation
        
        # State variables
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        calls = 0
        best_x = m.copy()
        best_f = func(m)
        calls += 1
        
        # Initial generation
        # To save time, we might skip a few generations if time is tight, but 500 budget is small.
        
        for gen in range(budget // lambda_):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
                
            # Eigen-decomposition of C for sampling
            eigval, eigvec = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigval = np.maximum(eigval, 1e-12)
            D = np.sqrt(eigval)
            
            # Sample lambda points
            Y = rng.standard_normal((lambda_, n))
            # Transform Y to have covariance C: B * D * Y
            # B is eigvec, D is sqrt(eigval)
            # X = m + sigma * (B * (D * Y.T)).T
            X = m + sigma * (eigvec * D) @ Y.T
            X = X.T # Shape (lambda_, n)
            
            # Evaluate
            F = np.empty(lambda_)
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    F[i] = F[i-1] if i > 0 else np.inf # Fallback, though we break soon
                    continue
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv @ (m - m_old))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - 2):
                sigma *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                
            # Update covariance
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - 2)
            hsig = float(hsig)
            pc = (1 - cc) * pc + hsig * cc * (m - m_old) / sigma
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * np.outer(ps, ps) + cmu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(X_sorted[:mu] - m, X_sorted[:mu] - m) / sigma**2, axis=0)
            
            # Update C_inv (expensive, but n=5 is small)
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(n)
                C = np.eye(n) # Reset if singular

        return best_x, best_f

# EVOLVE-BLOCK-END
