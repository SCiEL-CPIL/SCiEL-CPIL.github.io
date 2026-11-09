"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        damps = damps / (dim + 1)
        
        calls = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return 1e10
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initial evaluation
        f_mean = safe_func(x_mean)
        
        # Pre-compute Cholesky for C=I
        C_sqrt = np.eye(dim)
        
        # Main loop
        while calls < budget:
            # Generate population
            Y = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x = x_mean + sigma * y
                x = np.clip(x, -10, 10)
                Y.append(x)
            
            # Evaluate population
            F = np.array([safe_func(y) for y in Y])
            
            # If we ran out of budget during evaluation, break
            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = np.array([Y[i] for i in order])
            F_sorted = F[order]
            
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_sqrt @ (x_new_mean - x_mean))
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            pc = (1 - cc) * pc + cc * np.sqrt(cc * (2 - cc) * mueff) / sigma * (x_new_mean - x_mean)
            
            delta_sig = (np.linalg.norm(ps) / chi_n < 1.4 + 2 / (dim + 1))
            v1 = np.outer(ps, ps)
            v2 = np.outer(pc, pc)
            
            v3 = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (Y_sorted[i] - x_mean) / sigma
                v3 += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c1 * delta_sig * (1 - 2 / (dim + 1)) + cmu * delta_sig - c1 - cmu) * C \
                + c1 * (v1 + (1 - delta_sig) * v2 - C) \
                + cmu * v3
            
            C = (C + C.T) / 2
            
            x_mean = x_new_mean
            
            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
            
            # Check convergence
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
