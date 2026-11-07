import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls = 0

    def __call__(self, func):
        self.calls = 0
        self.best_f = np.inf
        self.best_x = None
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        # CMA-ES Parameters
        n = self.dim
        # Standard population size
        lam = int(4 + 3 * np.log(n))
        
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        mu = int(np.ceil(lam / 2))
        w_mu = w[:mu]
        w_mu /= np.sum(w_mu)
        mueff = 1 / np.sum(w_mu**2)
        
        # Adaptation parameters
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        # Initial sigma
        # The box is [-5, 5]. A good initial sigma is roughly range/sqrt(n) or similar.
        # Let's start with sigma = 2.0 as in the seed, but maybe larger for global search.
        
        remaining_budget = self.budget
        current_sigma = 2.0

        while remaining_budget > 0 and time.time() < deadline:
            # Initialize or continue
            if self.best_x is None:
                m = np.random.uniform(-5, 5, n)
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
            else:
                # m, C, p_sigma, p_c are from previous iteration (if not restarted)
                pass

            # Generate population
            # We need to be careful with the budget.
            # Each generation costs 'lam' evaluations.
            # If remaining_budget < lam, we might do a smaller step or just stop.
            # For simplicity, if remaining < lam, we do a (1+1) step or just stop.
            # But CMA-ES needs a population.
            # Let's check if we can afford a full generation.
            
            if remaining_budget >= lam:
                # Sample lam points
                # Use Cholesky or SVD for sqrt(C)
                # For n=5, eigendecomposition is cheap.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                # Generate standard normals
                Y = np.random.randn(lam, n)
                # Transform
                X = m[None, :] + current_sigma * (Y @ sqrtC.T)
                
                # Clip to box? The task says func clips. But we should stay in bounds to be efficient.
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(x) for x in X])
                self.calls += lam
                remaining_budget -= lam
                
                # Update best
                idx_best = np.argmin(F)
                if F[idx_best] < self.best_f:
                    self.best_f = F[idx_best]
                    self.best_x = X[idx_best].copy()

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]
                
                # Update mean
                m_new = m + current_sigma * np.sum(w_mu[:, None] * Y[:mu], axis=0)
                
                # Update covariance
                # Delta m
                delta_m = m_new - m
                m = m_new
                
                # Update cumulation paths
                # Normalized delta m
                norm_delta_m = np.linalg.norm(delta_m)
                if norm_delta_m > 1e-12:
                    delta_m_norm = delta_m / norm_delta_m
                else:
                    delta_m_norm = np.zeros(n)
                    
                # Update p_sigma
                p_sigma = (1 - cs) * p_sigma + cs * np.sqrt(mueff) * delta_m_norm
                
                # Update p_c
                # h_sigma: Heaviside function
                # Check if p_sigma is small enough
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2) else 0
                # Simplified h_sigma check for stability
                # Standard check: ||p_sigma|| / sqrt(1-(1-cs)^t) < (1.4 + 2/(n+1)) * sqrt(1-(1-cs)^t)
                # We don't track t, so use approximation or fixed threshold.
                # Let's use the standard approximation:
                threshold = (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2)
                if np.linalg.norm(p_sigma) < threshold:
                    h_sigma = 1
                else:
                    h_sigma = 0
                    
                p_c = (1 - cc) * p_c + h_sigma * cc * np.sqrt(mueff) * delta_m_norm
                
                # Update C
                C = (1 + c1 + cmu - c1) * C + c1 * (np.outer(delta_m_norm, delta_m_norm) - C)
                # Wait, the formula is:
                # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1-h_sigma) c1 (1-cc) C) + cmu * sum(w_i (y_i - y_w)(y_i - y_w)^T)
                
                # Let's stick to the simpler update if possible, or correct it.
                # Correct update:
                # C = (1 - c1 - cmu) * C
                # C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c1 * (1 - cc) * C)
                # C += cmu * np.sum([w_mu[i] * np.outer(Y[i], Y[i]) for i in range(mu)], axis=0)
                
                # Re-implementing C update correctly:
                C = (1 - c1 - cmu) * C
                C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c1 * (1 - cc) * C)
                # Sum of outer products of top mu Ys
                # Y is sorted, so Y[:mu] are the best
                for i in range(mu):
                    C += cmu * w_mu[i] * np.outer(Y[i], Y[i])
                    
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                # sigma = sigma * exp(cs / d_sigma * (||p_sigma|| / chi_n - 1))
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (25 * n**2))
                sigma_factor = np.exp(cs / (np.sqrt(n) + 1) * (np.linalg.norm(p_sigma) / chi_n - 1))
                current_sigma *= sigma_factor

                # Check for restart
                # If sigma is too small or too large
                if current_sigma < 1e-8 or current_sigma > 1e8:
                    # Restart
                    # Reset state for next iteration
                    # We will re-initialize m, C, etc. in the next loop
                    # But we need to mark it.
                    # Simplest: break and start a new "run" logic?
                    # Or just reset variables here.
                    # If we reset here, the next iteration will use the new m.
                    # But m was already updated.
                    # Let's just set a flag or reset variables.
                    # For simplicity, if we restart, we pick a new random m.
                    m = np.random.uniform(-5, 5, n)
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    current_sigma = 2.0 # Reset sigma
                    
            else:
                # Not enough budget for a full population
                # Do a (1+1) step or just stop
                # If we have < lam budget, we can do a few single steps.
                # Or just stop to be safe.
                # Let's do a simple (1+1) step if we have at least 1 eval.
                if remaining_budget >= 1:
                    y = np.clip(m + current_sigma * np.random.randn(n), -5, 5)
                    fy = func(y)
                    self.calls += 1
                    remaining_budget -= 1
                    if fy < self.best_f:
                        self.best_f = fy
                        self.best_x = y.copy()
                        m = y
                    else:
                        current_sigma *= 0.8
                else:
                    break

        return self.best_x, self.best_f
