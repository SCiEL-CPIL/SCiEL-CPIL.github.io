"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4 for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lambda_ = 4  # Population size
        mu = 2  # Number of selected solutions
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # State variables
        m = np.zeros(n)  # Mean (start at origin)
        sigma = 2.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        Cs = np.zeros(n)  # Cumulation for sigma
        pc = np.zeros(n)  # Cumulation for C
        B, d = np.linalg.eigh(C)  # Eigendecomposition
        
        # Evaluate initial point
        fx_init = func(m.copy())
        best_x = m.copy()
        best_f = fx_init
        evals = 1
        
        # Precompute weights for selection
        # We need to sort and take top mu
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Generate lambda samples
            samples = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Transform using eigendecomposition
                x_sample = m + sigma * B @ (d ** 0.5) @ z
                x_sample = np.clip(x_sample, -5, 5)
                samples.append(x_sample)
            
            # Evaluate all samples
            fvals = []
            for x_s in samples:
                f_s = func(x_s)
                fvals.append(f_s)
                evals += 1
                
                if f_s < best_f:
                    best_f = f_s
                    best_x = x_s.copy()
            
            # Sort by fitness (ascending)
            sorted_idx = np.argsort(fvals)
            selected_idx = sorted_idx[:mu]
            selected_samples = [samples[i] for i in selected_idx]
            
            # Update mean using weighted recombination
            m_old = m.copy()
            m_new = sum(w[i] * selected_samples[i] for i in range(mu))
            
            # Update cumulation for sigma
            z_bar = (B.T @ (m_new - m_old) / sigma)
            Cs = (1 - c_sigma) * Cs + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_bar
            
            # Update sigma (CSA - Cumulative Step-size Adaptation)
            norm_cs = np.linalg.norm(Cs)
            sigma *= np.exp(c_sigma / chi_n * (norm_cs / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update cumulation for C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * z_bar
            
            # Update covariance matrix
            delta_hs = (1 - c_c) ** 2 - c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = sum(w[i] * np.outer(selected_samples[i] - m_old, 
                                           selected_samples[i] - m_old) 
                        for i in range(mu)) / sigma ** 2
            
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update eigendecomposition
            B, d = np.linalg.eigh(C)
            # Ensure d is non-negative
            d = np.maximum(d, 1e-12)
            
            m = m_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
