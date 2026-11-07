"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for Katsuura (f23)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, fx0, budget, deadline, rng):
        """CMA-ES variant with adaptive sigma and covariance matrix."""
        dim = self.dim
        x = x0.copy()
        fx = fx0
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES parameters
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2.0 / (dim + 1.3) - (dim + 2) / (dim + 1.3)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3) - 1) - 1) + cs
        cc = (4 + dim / dim) / (dim + 4)
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        calls = 1
        while calls < budget and time.time() < deadline:
            # Sample lambda = 4 for dim=5 (small population)
            lam = max(4, 4 + int(3 * np.log(dim)))
            y = [rng.multivariate_normal(np.zeros(dim), C) * sigma for _ in range(lam)]
            y = np.array(y)
            # Ensure y is properly shaped: lam x dim
            if y.shape[0] != lam:
                y = y.reshape(lam, dim)
            
            # Generate candidates
            candidates = x + y
            # Clip to box
            candidates = np.clip(candidates, -5, 5)
            
            # Evaluate
            vals = np.array([func(candidates[i]) for i in range(lam)])
            calls += lam
            
            if calls > budget:
                # Adjust: we may have exceeded budget, but keep going until deadline
                pass
            
            # Sort by fitness
            order = np.argsort(vals)
            mu = lam // 2  # top half
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            
            # Update mean
            y_sorted = y[order[:mu]]
            x_new = x + np.sum(w[:, None] * y_sorted, axis=0)
            x_new = np.clip(x_new, -5, 5)
            fx_new = vals[order[0]]
            
            # Update evolution paths
            # Need y in C^{-1/2} space; approximate with y/sigma for simplicity
            y_sigma = y_sorted / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C @ (x_new - x)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - budget)))) < (1.4 + 2 / (dim + 1)) * chiN
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new - x) / sigma
            
            # Update covariance
            delta = np.zeros(dim)
            if hsig:
                delta = (cc - 1) * pc @ pc.T
            C = (1 + c1 + cmu - c1) * C - c1 * C + \
                cmu / mueff * np.sum(w[:, None] * (y_sigma[:, None] * y_sigma[:, None]).T, axis=0)
            # Simpler: C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + delta) + cmu * sum(w_i * y_i y_i^T)
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + delta) + \
                cmu * np.sum(w[:, None] * (y_sigma[:, None] * y_sigma[:, None]).T, axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            
            x = x_new
            fx = fx_new
            
            # If sigma is too small, stop
            if sigma < 1e-10:
                break

        return x, fx, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 300s limit
        
        best_x = None
        best_fx = np.inf
        calls_made = 0
        
        # Multi-start: try several initial points
        # First, a few random starts
        n_starts = 5
        for i in range(n_starts):
            if time.time() > deadline - 10:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            calls_made += 1
            
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
            
            # Run CMA-ES from this start with a fraction of the budget
            remaining = self.budget - calls_made
            if remaining <= 0:
                break
            # Give each start about 1/5 of the budget, but not more than remaining
            sub_budget = max(20, min(remaining, self.budget // n_starts))
            
            x_res, fx_res, calls_used = self._cma_es(func, x0, fx0, sub_budget, deadline, rng)
            calls_made += calls_used
            
            if fx_res < best_fx:
                best_fx = fx_res
                best_x = x_res.copy()
        
        # If we have budget left, do a final polish with Nelder-Mead or similar
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            # Use scipy.optimize.minimize with Nelder-Mead for local polish
            try:
                from scipy.optimize import minimize
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
