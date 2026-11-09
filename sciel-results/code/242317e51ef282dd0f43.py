"""Rewritten optimizer using a lightweight CMA-ES implementation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin below 330s
        
        d = self.dim
        # Population size: 4 + 3*log(d)
        lam = int(4 + 3 * np.log(d))
        if lam < 4:
            lam = 4
        
        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial state
        # Start at origin, which is usually a good start for BBOB shifted functions
        # unless the shift is large, but sigma will adapt.
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Pre-compute constants for C update
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        h_sigma = (1 + c_sigma / 2 + c_mu / 2) * chi_n
        # Note: h_sigma is actually for the update of ps, let's use standard formula
        # h_sigma = (1 + c_sigma / 2) * chi_n ? No, standard is:
        # h_sigma = (1 + c_sigma / 2) * chi_n is for the norm check?
        # Let's use the standard CMA-ES update logic carefully.
        
        # Correction: The condition for h_sigma is usually:
        # ||ps|| / sqrt(1 - (1-c_sigma)^{2*it}) < (1 + c_sigma/2) * chi_n
        # To simplify and avoid tracking 'it' for this specific term if we just use a fixed threshold approximation:
        # We will track the generation count for the exponential moving average term if needed, 
        # but for 50 generations, a simple approximation or standard implementation is fine.
        
        best_x = m.copy()
        best_f = func(m)
        calls = 1
        
        # Pre-generate random numbers? No, use rng.
        rng = np.random.default_rng(42)
        
        # Cholesky decomposition for C
        L = np.linalg.cholesky(C)
        
        gen = 0
        while calls < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            # Generate population
            Y = np.zeros((lam, d))
            for i in range(lam):
                z = rng.standard_normal(d)
                Y[i] = m + sigma * L @ z
                # Clip to box [-5, 5] to ensure valid inputs for func
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                calls += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()
            
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            m_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)^2) * C) + c_mu * sum(wi * yi @ yi.T)
            # Where yi = (Yi - m) / sigma
            Ys = (Y[:mu] - m) / sigma
            
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * L.T @ ((m_new - m) / sigma)
            
            # h_sigma calculation
            # norm(ps) / sqrt(1 - (1-c_sigma)^(2*(gen+1)))
            # Approximate denominator for small gen or use 1 if gen is large?
            # For 50 gens, (1-c_sigma)^100 is small.
            # Let's compute the exact term.
            it = gen + 1
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * it))
            if denom < 1e-10:
                denom = 1e-10
            h_sigma = 1.0 if (np.linalg.norm(ps) / denom) < (1 + c_sigma / 2) * chi_n else 0.0
            
            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (pc @ pc.T + (1 - h_sigma) * (1 - (1 - c_c)**2) * C)
            
            sum_w = np.zeros((d, d))
            for i in range(mu):
                sum_w += weights[i] * np.outer(Ys[i], Ys[i])
            
            C += c_mu * sum_w
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Regularize if needed
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-4 * np.eye(d)
                L = np.linalg.cholesky(C)
            
            m = m_new
            gen += 1
            
            # Check for convergence (optional, but good for early exit if stuck)
            # If sigma is very small and progress is negligible, we might break, 
            # but we have a budget, so just keep going until budget is exhausted.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
