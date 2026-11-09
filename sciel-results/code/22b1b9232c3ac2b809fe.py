"""CMA-ES optimizer with deadline guard for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin: 330s limit, leave 50s buffer
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Standard recommendations for dim=5
        lambda_ = max(4, int(4 + 3 * np.log(dim))) # lambda = 9 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial state
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        sigma = 2.0 # Initial step size, box is [-5, 5]^5, width 10. 2.0 is reasonable.
        
        # Covariance matrix C and its square root representation
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x.copy()
        best_f = float('inf')
        
        # We need to make exactly 500 calls.
        # CMA-ES usually uses a population of lambda.
        # With budget 500 and lambda=9, we can do ~55 generations.
        
        calls_made = 0
        
        def make_call(x_eval):
            nonlocal calls_made
            # Clip to box
            x_eval = np.clip(x_eval, -5, 5)
            f_val = func(x_eval)
            calls_made += 1
            return f_val, x_eval

        # Initial evaluation
        f_init, x_init = make_call(x)
        best_f = f_init
        best_x = x_init.copy()
        
        for gen in range(1, 200): # Upper bound on generations, will break by budget or time
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Generate population
            pops = []
            fitnesses = []
            
            # Number of individuals to generate this generation
            # Must not exceed remaining budget
            remaining = budget - calls_made
            n_gen = min(lambda_, remaining)
            
            if n_gen == 0:
                break
            
            for i in range(n_gen):
                # Sample
                z = rng.standard_normal(dim)
                # Transform using B and D
                y = B @ (D * z)
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new, x_evaluated = make_call(x_new)
                pops.append(x_evaluated)
                fitnesses.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_evaluated.copy()
                    
            if calls_made >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fitnesses)
            pops_sorted = [pops[i] for i in idx]
            fitnesses_sorted = [fitnesses[i] for i in idx]
            
            # Compute weighted mean
            w = weights[:len(pops_sorted)]
            if len(w) < mu:
                 # If we didn't get a full population (shouldn't happen if n_gen >= mu), handle gracefully
                 w = np.ones(len(pops_sorted)) / len(pops_sorted)
            
            m = np.sum([w[i] * pops_sorted[i] for i in range(len(w))], axis=0)
            
            # Update sigma and paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D * (m - x))
            
            # h_sigma check
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (len(w) + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m - x)
            
            # Update C
            # Rank-1 update
            c1 = c_1
            cmu = c_mu
            rank1 = np.outer(p_c, p_c)
            rankmu = np.zeros((dim, dim))
            for i in range(len(w)):
                y_i = (pops_sorted[i] - x) / sigma
                rankmu += w[i] * np.outer(y_i, y_i)
                
            C = (1 + c1 * h_sigma * (1 - h_sigma) - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            
            # Update B and D
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update x and sigma
            x = m
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Check for convergence
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
