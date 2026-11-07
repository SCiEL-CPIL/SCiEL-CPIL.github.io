"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions.
    
    Uses a standard CMA-ES implementation with adaptive population size and 
    step size. This approach is generally superior to (1+1)-ES for 
    non-separable functions like Schaffers F7, especially with hidden rotations.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # Safety margin vs 330s limit

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of selected individuals
        
        # Initial state
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        
        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        
        # Control parameters
        cs = 1 + 2 * min(1.0 / (lam - 2), 0.5 / (dim + 2))
        ds = 1 + 0.5 * cs + 2 * max(0, min(1.0 / (lam - 2) - 1, 0.5 / (dim + 2)))
        cc = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)
        cmu = min(1 - cc, 2 * (np.sqrt(2 * dim) - 2) / ((dim + 2) ** 2 + lam - 2))
        hsig = (1 + 2 / (dim + 1)) * np.sqrt((dim + 1 / dim - 2) / (dim + 1 / dim))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        n_calls = 0
        x_best = mean.copy()
        f_best = np.inf
        gen = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        # Main Loop
        while n_calls < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            # If not, we might still do it if we are close, but usually 
            # we stop if we can't complete a generation to maintain CMA-ES integrity.
            # However, for AOCC, even partial progress counts. 
            # But CMA-ES updates are generation-based. 
            # Let's just run as many full generations as possible.
            
            # 1. Sampling
            Z = np.random.randn(lam, dim)
            Y = np.dot(Z, np.outer(D, B.T))
            X = mean + sigma * Y
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if n_calls >= budget or time.time() > deadline:
                    break
                F[i] = func(X[i])
                record(X[i], F[i])
            
            if n_calls >= budget or time.time() > deadline:
                break
            
            # 2. Selection
            idx = np.argsort(F)
            selected_idx = idx[:mu]
            Y_selected = Y[selected_idx]
            X_selected = X[selected_idx]
            
            # 3. Update mean
            m_new = np.sum(weights[:, None] * X_selected, axis=0)
            
            # 4. Update step size
            delta = (m_new - mean) / sigma
            delta_y = np.dot(B.T, delta / D)
            
            # h_sigma calculation
            # norm(p_sigma) / sqrt(1 - (1-cs)^(2/gen))
            # We need gen to be > 0
            gen += 1
            if gen > 0:
                decay = 1 - (1 - cs) ** (2 / gen)
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if norm_p_sigma / np.sqrt(decay) < (1 + 2 / (dim + 1)) * hsig else 0.0
            else:
                h_sigma = 1.0
                
            p_sigma = (1 - cs) * p_sigma + h_sigma * np.sqrt(mueff) * delta_y
            sigma *= np.exp(ds / cs * (np.linalg.norm(p_sigma) / hsig - 1))
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # 5. Update covariance matrix
            # h_c calculation
            # norm(p_c) / sqrt(1 - (1-cc)^(2/gen))
            if gen > 0:
                decay_c = 1 - (1 - cc) ** (2 / gen)
                norm_p_c = np.linalg.norm(p_c)
                h_c = 1.0 if norm_p_c / np.sqrt(decay_c) < (1 + 2 / (dim + 1)) * hsig else 0.0
            else:
                h_c = 1.0
                
            p_c = (1 - cc) * p_c + h_c * np.sqrt(mueff) * delta_y
            
            # Update C
            # C = (1 - mu_c) * C + mu_c * (p_c p_c^T + h_c (1-h_c) cc C + sum w_i (y_i y_i^T))
            # mu_c = cmu + (1 - cmu) * (1 - h_c^2)
            mu_c = cmu + (1 - cmu) * (1 - h_c ** 2)
            
            # Sum of outer products of selected Y
            # Y_selected is (mu, dim)
            # We need sum_i w_i * Y_selected[i] * Y_selected[i].T
            # This can be computed as (Y_selected * weights[:, None]).T @ Y_selected
            YW = Y_selected * weights[:, None]
            rank_mu = np.dot(YW.T, Y_selected)
            
            C = (1 - mu_c) * C + mu_c * (np.outer(p_c, p_c) + h_c * (1 - h_c) * cc * C + rank_mu)
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update B, D
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            
            # Update mean
            mean = m_new
            
            # Check for convergence or stagnation
            # If sigma is too small or variance is too low, we might restart
            # But for now, we just continue.
            
        return x_best, f_best
# EVOLVE-BLOCK-END
