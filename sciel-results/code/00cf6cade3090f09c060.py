"""Robust CMA-ES for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust CMA-ES implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial state
        x = np.zeros(dim)  # Start at origin, usually good for shifted/rotated box if centered
        # Actually, random start is safer if shift is unknown.
        x = rng.uniform(-4, 4, dim)
        
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x.copy()
        best_fx = func(x)
        calls_made = 1
        
        # Pre-compute expected norm of standard normal
        E_norm = np.sqrt(np.pi * dim / 4) # Approximation for small dim, or use exact
        # For dim=5, E||N(0,I)|| is approx 2.13
        
        while calls_made < budget and time.time() < deadline:
            # Check if we can afford a full generation
            if budget - calls_made < lambda_:
                break
            
            # Generate candidates
            z = rng.standard_normal((lambda_, dim))
            
            # Transform using current C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: restart with identity covariance if C is broken
                C = np.eye(dim)
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                L = np.eye(dim)
                
            ys = x + sigma * (L @ z.T).T
            
            # Evaluate
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                # Clip to box for evaluation? The func clips, but we can clip ourselves to be safe for statistics
                # However, CMA-ES assumes unbounded. Clipping can distort the distribution.
                # We rely on func's clipping as per contract.
                fvals[i] = func(ys[i])
                calls_made += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = ys[i].copy()
                
                if time.time() > deadline:
                    break
            
            if calls_made >= budget or time.time() >= deadline:
                break
            
            # Sort
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]
            
            # New mean
            x_new = np.sum(ys[:mu] * weights[:, None], axis=0)
            
            # Update paths
            delta = (x_new - x) / sigma
            
            # C^{-1/2}
            try:
                w, V = np.linalg.eigh(C)
                w = np.clip(w, 1e-12, None)
                C_inv_sqrt = V @ np.diag(1 / np.sqrt(w)) @ V.T
            except Exception:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta
            
            # h_sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * E_norm else 0.0
            
            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = ys[i] - x
                rank_mu += weights[i] * np.outer(diff, diff)
                
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1.0))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
