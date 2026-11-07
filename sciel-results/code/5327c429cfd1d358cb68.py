"""Combined CMA-ES with Nelder-Mead polish for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive covariance followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Reserve budget for Nelder-Mead polish
        polish_budget = min(80, budget // 5)
        es_budget = budget - polish_budget

        # CMA-ES parameters
        lam = 9  # lambda
        mu = lam // 2  # mu
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / (weights[0] * mu)  # effective population size
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        # Initial covariance: identity
        C = np.eye(dim)
        # Initial step size
        sigma = 2.0
        
        # Precompute Cholesky decomposition
        L = np.linalg.cholesky(C)
        
        # CMA-ES parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        
        # Initial evaluation
        fx = func(mean)
        best_x = mean.copy()
        best_fx = fx
        calls_made = 1
        
        # Track best
        while calls_made < es_budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                if calls_made >= es_budget or time.time() >= deadline:
                    break
                fy = func(y)
                calls_made += 1
                offspring_f.append(fy)
            
            if len(offspring_f) < lam:
                # Pad with current mean value if budget ran out
                while len(offspring_f) < lam:
                    offspring_f.append(fx)
                    offspring.append(mean.copy())
            
            # Sort by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]
            
            # Update best
            if sorted_f[0] < best_fx:
                best_fx = sorted_f[0]
                best_x = sorted_offspring[0].copy()
            
            # Update mean
            new_mean = sum(weights[i] * sorted_offspring[i] for i in range(mu))
            
            # Update evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(L) * (L.T @ (new_mean - mean) / sigma)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam)))) < (1.4 + 2 / (dim + 1)) * chiN
            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff * cc) * np.outer(new_mean - mean, new_mean - mean)
            
            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - c1 - cmu + delta_hsig) * C + c1 * np.outer(new_mean - mean, new_mean - mean)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (sorted_offspring[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C += cmu * rank_mu
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update step size
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            if sigma > 100:
                sigma = 100
            elif sigma < 1e-15:
                sigma = 1e-15
            
            mean = new_mean
            L = np.linalg.cholesky(C)
            
            # Check convergence
            if np.linalg.norm(mean - best_x) < 1e-10 and sigma < 1e-8:
                break

        # Phase 2: Nelder-Mead Polish
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining_budget:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                # Clip to box to ensure validity
                z_clipped = np.clip(z, -5, 5)
                return func(z_clipped)
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
