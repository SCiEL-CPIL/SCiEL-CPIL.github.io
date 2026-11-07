"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cm = (2 + min(1 - 1 / cc, (mueff - 2 + 1 / mueff) / (dim + 2))) / (dim + 1 + cc + 2 * min(1 - 1 / cc, (mueff - 2 + 1 / mueff) / (dim + 2)))
        
        chisq = dim * (1 - 1 / (4 * dim) - 1 / (25 * dim ** 2))
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Restart logic: Try a few different initializations if the first one stalls
        # Start at 0 is good for Sphere with shift in [-4, 4]
        inits = [np.zeros(dim)]
        # Add a few random inits if budget allows
        rng = np.random.default_rng(42)
        for _ in range(3):
            inits.append(rng.uniform(-4, 4, dim))

        for x_init in inits:
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            x = x_init.copy()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if calls_made >= total_budget:
                break

            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            D = np.ones(dim)
            B = np.eye(dim)
            
            stagnation_count = 0
            prev_best_f = fx
            
            while calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Sample
                offspring = []
                for k in range(lam):
                    z = np.random.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    offspring.append(y)
                
                # Evaluate
                offspring_f = []
                for y in offspring:
                    fy = func(y)
                    calls_made += 1
                    offspring_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                # Sort
                indices = np.argsort(offspring_f)
                offspring = [offspring[i] for i in indices]
                offspring_f = [offspring_f[i] for i in indices]
                
                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (offspring[i] - x)
                x_new = x + y_w
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * (B.T @ (y_w / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((lam / 2 + 1) / (dim + 1)):
                    sigma *= np.exp(cs * (np.linalg.norm(ps) / np.sqrt(chisq) - 1) / ds)
                    if sigma > 10.0: sigma = 10.0
                    if sigma < 1e-10: sigma = 1e-10
                    
                # Update covariance
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    vi = (offspring[i] - x) / sigma
                    rank_mu += weights[i] * np.outer(vi, vi)
                
                rank_one = np.outer(ps, ps)
                delta_hs = int(np.linalg.norm(ps) / np.sqrt(chisq) < (1.4 + 2 / (dim + 1)) * np.sqrt((lam / 2 + 1) / (dim + 1)))
                
                coeff_c = 1 - cm + cm * (1 - delta_hs) * (cc * (2 - cc) - 1)
                coeff_r1 = cm * (1 + cc * (2 - cc))
                C = coeff_c * C + coeff_r1 * rank_one
                
                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
                
                x = x_new
                
                # Stagnation check
                if offspring_f[0] < prev_best_f - 1e-10:
                    prev_best_f = offspring_f[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Restart if stagnation is high
                if stagnation_count > 10:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
