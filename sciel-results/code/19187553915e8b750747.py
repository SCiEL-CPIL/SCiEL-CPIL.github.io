"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        lambda_ = max(10, lambda_)
        mu = lambda_ // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        
        fx = func(mean)
        calls_made = 1
        best_x = mean.copy()
        best_f = fx

        def remaining():
            return budget - calls_made

        def time_ok():
            return time.time() < deadline

        while remaining() >= lambda_ and time_ok():
            ys = []
            fs = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # x = mean + sigma * B * (D * z)
                y = mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            ys = np.array(ys)
            fs = np.array(fs)
            
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            ydiff = (ys[:mu] - mean_old) / sigma
            delta_sigma = np.linalg.norm(mean - mean_old) / (sigma * np.sqrt(2))
            
            pc = (1 - cs) * pc + cs * delta_sigma * (mean - mean_old) / (sigma * np.sqrt(2) + 1e-10)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * ds) * (mean - mean_old) / sigma
            
            c_sigma = cc * (1 - cc) + cs * (1 - cs)
            rank_one = np.outer(ps, ps)
            rank_mu = np.sum(weights[:, None, None] * np.outer(ydiff, ydiff), axis=0)
            
            C = (1 + c1 * c_sigma - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, 1e10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                pass

            if sigma < 1e-10:
                break

        if remaining() > 20 and time_ok():
            try:
                from scipy.optimize import minimize
                def wrapper(x):
                    nonlocal calls_made
                    if calls_made >= budget:
                        return best_f
                    val = func(x)
                    calls_made += 1
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val
                minimize(wrapper, best_x, method="Nelder-Mead", options={"maxiter": 50, "xatol": 1e-8, "fatol": 1e-10})
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
