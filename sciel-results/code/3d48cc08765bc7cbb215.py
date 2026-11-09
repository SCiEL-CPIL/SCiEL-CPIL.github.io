"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using Bayesian Optimization with GP surrogate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """
    Solver using Bayesian Optimization with Gaussian Process surrogate and Expected Improvement acquisition.
    
    Strategy:
    1. Initialize with a small random set of points.
    2. Iteratively select the next point by maximizing Expected Improvement (EI).
    3. Update the GP model with new observations.
    4. Use a deadline guard to ensure we finish before the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Bayesian Optimization parameters
        rng = np.random.default_rng(42)
        
        # Initial points: mix of center and random
        n_init = 10
        X_init = np.zeros((n_init, dim))
        X_init[0] = np.zeros(dim)  # center
        X_init[1:] = rng.uniform(-4, 4, size=(n_init - 1, dim))
        
        f_init = np.zeros(n_init)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            f_init[i] = counting_func(X_init[i])
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Trim to actual number of evaluations
        n_actual = min(n_init, calls)
        X = X_init[:n_actual]
        f = f_init[:n_actual]
        
        # GP kernel parameters (Matern 5/2)
        # Length scales: start with moderate values
        length_scales = np.ones(dim) * 1.5
        signal_var = 1.0
        noise_var = 1e-6
        
        # Number of candidate points for acquisition
        n_candidates = 500
        
        # EI exploration parameter
        xi = 0.01
        
        # Maximum iterations for BO
        max_bo_iters = budget - n_actual
        
        # Precompute candidate points (Halton-like quasi-random)
        # Use Sobol-like or Halton for better coverage
        def halton_sequence(n, dim):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    # Halton base: use (j+1)-th prime
                    base = _halton_base(j + 1)
                    f = 1.0
                    r = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i, j] = r
            return points
        
        # Generate candidate points in the box [-5, 5]^dim
        halton_unit = halton_sequence(n_candidates, dim)
        X_candidates = -5 + 10 * halton_unit
        
        # BO loop
        for bo_iter in range(max_bo_iters):
            if calls >= budget or time.time() > deadline:
                break
            
            # Update GP with current data
            # Normalize f for numerical stability
            f_mean = np.mean(f)
            f_std = np.std(f) + 1e-10
            f_norm = (f - f_mean) / f_std
            
            # Compute K(X, X)
            K = _matern52_kernel(X, X, length_scales, signal_var) + noise_var * np.eye(len(X))
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
            except:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
            
            # Compute alpha = K^{-1} f
            try:
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_norm))
            except:
                alpha = np.linalg.solve(K + 1e-6 * np.eye(len(X)), f_norm)
            
            # Compute K(X_candidates, X)
            K_s = _matern52_kernel(X_candidates, X, length_scales, signal_var)
            
            # Predictive mean and variance
            f_pred_mean = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            f_pred_var = signal_var - np.sum(v**2, axis=0)
            f_pred_var = np.maximum(f_pred_var, 1e-10)
            
            # Denormalize predictions
            f_pred_mean = f_pred_mean * f_std + f_mean
            f_pred_std = np.sqrt(f_pred_var) * f_std
            
            # Expected Improvement acquisition
            best_f_current = np.min(f)
            improvement = best_f_current - f_pred_mean
            z = improvement / (f_pred_std + 1e-10)
            EI = improvement * norm.cdf(z) + f_pred_std * norm.pdf(z)
            
            # Add small random perturbation to avoid determinism
            EI += rng.normal(0, 0.001, size=len(EI))
            
            # Select point with maximum EI
            idx_max = np.argmax(EI)
            x_next = X_candidates[idx_max]
            
            # Evaluate
            f_next = counting_func(x_next)
            
            # Add to data
            X = np.vstack([X, x_next])
            f = np.append(f, f_next)
            
            # Adaptively adjust length scales based on convergence
            # If improvement is small, reduce exploration
            if bo_iter > 10:
                recent_improvements = np.diff(f[-20:]) if len(f) >= 20 else np.diff(f)
                if len(recent_improvements) > 0 and np.mean(np.abs(recent_improvements)) < 0.01 * (np.abs(best_f) + 1):
                    # Stagnation: shrink length scales to focus locally
                    length_scales *= 0.9
                    xi *= 0.8
        
        return best_x, best_f


def _halton_base(n):
    """Return the n-th prime number (1-indexed)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
              157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233]
    if n <= len(primes):
        return primes[n - 1]
    # Fallback for larger n (should not be needed for dim=5)
    candidate = primes[-1] + 2
    while True:
        if all(candidate % p != 0 for p in range(2, int(np.sqrt(candidate)) + 1)):
            return candidate
        candidate += 2


def _matern52_kernel(X1, X2, length_scales, signal_var):
    """Compute Matern 5/2 kernel between X1 and X2."""
    # Scale inputs
    X1_scaled = X1 / length_scales
    X2_scaled = X2 / length_scales
    
    # Compute squared distances
    sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * X1_scaled @ X2_scaled.T
    sq_dists = np.maximum(sq_dists, 0)
    dists = np.sqrt(sq_dists)
    
    # Matern 5/2 kernel
    sqrt3_dists = np.sqrt(3) * dists
    K = signal_var * (1 + sqrt3_dists + sqrt3_dists**2 / 3) * np.exp(-sqrt3_dists)
    
    return K
# EVOLVE-BLOCK-END
