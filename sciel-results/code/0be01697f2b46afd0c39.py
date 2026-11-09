import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        d = self.dim
        # Population size
        lambda_pop = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        alpha_mu = 0.5 * (1 + 1.0 / d)
        c_sigma = (2 + 2.0 / (d + 1)) / (d + 4 + 2.0 / (d + 1))
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (alpha_mu - 1) / ((d + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d**2))
        
        # Initial state
        mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        rng = np.random.default_rng(42)
        
        while calls < self.budget and (time.time() - start_time) < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)
                
            pop = np.zeros((lambda_pop, d))
            for i in range(lambda_pop):
                z = rng.standard_normal(d)
                pop[i] = mean + sigma * L.dot(z)
                
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls += lambda_pop
            
            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()
                
            # Sort
            order = np.argsort(fvals)
            y = pop[order]
            
            # Recombination
            y_w = np.dot(weights[:mu], y[:mu])
            mean_old = mean.copy()
            mean = mean + c_1 * (y_w - mean) + c_mu * (np.dot(weights[:mu], y[:mu]) - y_w)
            
            # Evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T.dot(y_w - mean_old))
            
            # h_sigma check
            remaining = self.budget - calls
            if remaining > 0:
                h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(remaining / c_sigma)) < (1.4 + 2 / (d + 1)) * chi_n)
            else:
                h_sig = True
                
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - mean_old) / sigma
            
            # Update Covariance
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            diffs = y[:mu] - mean_old
            C += c_mu * (diffs.T * weights[:mu]) @ diffs
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Convergence check
            if np.all(np.abs(pop - mean) < 1e-12):
                break

        return best_x, best_f
