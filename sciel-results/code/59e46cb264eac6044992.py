"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES population-based optimizer for f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim)) for dim=5 -> 4+3*1.609=4+4=8
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2  # number of parents
        
        # Initialize mean at origin
        mean = np.zeros(dim)
        sigma = 1.0
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        
        # Precompute weights for recombination
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters (1/5 success rule adapted for CMA-ES)
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - 1 / ds)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        # Evolution path for step size adaptation
        p_sigma = np.zeros(dim)
        # Evolution paths for covariance adaptation
        p_c = np.zeros(dim)
        p_sigma_norm = 0.0
        
        # Best tracking
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1

        def remaining():
            return budget - calls_made

        def time_ok():
            return time.time() < deadline

        # Main CMA-ES loop
        while remaining() > lambda_ and time_ok():
            # Generate lambda_ samples
            # Compute eigenvalue decomposition for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            
            # Generate lambda_ random vectors
            z = rng.standard_normal((lambda_, dim))
            # Transform to sample space
            samples = mean + sigma * (eigvecs @ (z * sqrt_eigvals).T).T
            samples = np.clip(samples, -5, 5)
            
            # Evaluate all samples
            fvals = np.array([func(s) for s in samples])
            calls_made += lambda_
            
            # Sort by fitness
            idx = np.argsort(fvals)
            x_selected = samples[idx[:mu]]
            f_selected = fvals[idx[:mu]]
            
            # Update best
            if fvals.min() < best_f:
                best_f = fvals.min()
                best_x = samples[idx[0]].copy()
            
            # Weighted recombination
            new_mean = np.sum(weights[:, np.newaxis] * x_selected, axis=0)
            
            # Update evolution path for step size
            # Compute inverse square root of C for transforming step
            inv_sqrt_C = (eigvecs * (1.0 / sqrt_eigvals)) @ eigvecs.T
            step = inv_sqrt_C @ (new_mean - mean)
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * step
            p_sigma_norm = np.linalg.norm(p_sigma)
            
            # Hessian correction (delta_h)
            hsig = (p_sigma_norm / np.sqrt(1 - (1 - cs) ** (2 * (calls_made / (lambda_ * ds))))) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (calls_made / (lambda_ * ds))))
            hsig = float(hsig)
            
            # Update step size
            sigma *= np.exp(cs / ds * (p_sigma_norm / np.sqrt(mueff) - 1))
            
            # Update evolution path for covariance
            p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mueff) * step
            
            # Rank-mu update
            outer = np.zeros((dim, dim))
            for i in range(mu):
                diff = x_selected[i] - mean
                outer += weights[i] * np.outer(diff, diff)
            
            # Update covariance matrix
            C = (1 + c1 * (hsig * (2 - c1) - 1) + cmu) * C \
                - c1 * np.outer(p_c, p_c) \
                - cmu / mueff * outer
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals_new, _ = np.linalg.eigh(C)
            if np.min(eigvals_new) < 1e-12:
                C += 1e-12 * np.eye(dim)
            
            # Update mean
            mean = new_mean
            
            # Early termination check
            if sigma < 1e-8:
                break
        
        # Final polish with Nelder-Mead if budget remains
        if remaining() > 10 and time_ok():
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

                res = minimize(
                    wrapper,
                    best_x,
                    method="Nelder-Mead",
                    options={"maxiter": 50, "xatol": 1e-8, "fatol": 1e-10},
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
