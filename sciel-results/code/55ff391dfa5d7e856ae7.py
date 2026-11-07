"""Rewritten optimizer for BBOB-style Weierstrass f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: evaluate center and a few random points to find a good start
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-4, 4, dim))
        
        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        # CMA-ES parameters
        n = dim
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Initial sigma
        sigma = 1.0
        # Weights for recombination
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / n) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State variables
        x_mean = best_x.copy()
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # Precompute B and D for C^{1/2}
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 0))
        
        # Track calls
        calls_made = len(candidates)
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.empty((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                pop[i] = x_mean + sigma * (B @ (D * z))
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    fvals[i:] = np.inf
                    break
                fvals[i] = func(pop[i])
                calls_made += 1
            
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update step-size control
            y_mean = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made / (lambda_ * c_sigma)))) < (1.4 + 2 / (n + 1)) * chi_n
            if hsig:
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Update covariance matrix
            delta = (hsig - 1 + c_1) * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = (pop[i] - x_mean) / sigma
                delta += c_mu * weights[i] * np.outer(y_i, y_i)
            C = (1 + c_1 * (2 - c_1) + c_mu - c_1) * C + delta
            
            # Recompute B and D
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 0))
            
            x_mean = x_mean_new
        
        return best_x, best_f
# EVOLVE-BLOCK-END
