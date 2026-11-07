"""Bayesian Optimization with RBF surrogate + periodic local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import Rbf
import random


class Optimizer:
    """
    Bayesian Optimization with RBF surrogate for Schaffers F7.
    
    Strategy:
    1. Use a Sobol-like quasi-random sequence for initial exploration.
    2. Fit an RBF surrogate to observed data.
    3. Use Expected Improvement (EI) acquisition to select next points.
    4. Periodically run L-BFGS-B polish on the best point found.
    
    This exploits the smooth structure of Schaffers F7 more efficiently
    than population-based methods, focusing evaluations on promising regions.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0
        
        # Data storage for surrogate
        X_data = []
        f_data = []
        
        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            X_data.append(x.copy())
            f_data.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Initial exploration with quasi-random points
        # Use stratified sampling for better coverage
        n_initial = min(50, budget // 4)
        
        # Generate stratified random points
        rng = np.random.RandomState(42)
        for i in range(n_initial):
            if time.time() >= deadline or call_count >= budget:
                break
            # Stratified sampling in 5D
            x = np.array([rng.uniform(-5.0, 5.0) for _ in range(dim)])
            evaluate(x)
        
        # Phase 2: Bayesian Optimization with RBF surrogate
        polish_interval = 30  # Polish every 30 evaluations
        last_polish = 0
        
        while call_count < budget and time.time() < deadline:
            # Check if we should polish
            if call_count - last_polish >= polish_interval and best_x is not None:
                last_polish = call_count
                remaining = budget - call_count
                if remaining > 15:
                    try:
                        result = minimize(
                            evaluate,
                            best_x,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': min(30, remaining - 5), 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x
                    except Exception:
                        pass
            
            if call_count >= budget or time.time() >= deadline:
                break
            
            # Fit RBF surrogate if we have enough data
            if len(X_data) < 10:
                # Not enough data, use random point
                x_new = np.array([rng.uniform(-5.0, 5.0) for _ in range(dim)])
                evaluate(x_new)
                continue
            
            try:
                X = np.array(X_data)
                f = np.array(f_data)
                
                # Fit RBF surrogate
                # Use a subset if too many points
                if len(X) > 50:
                    idx = rng.choice(len(X), 50, replace=False)
                    X_fit = X[idx]
                    f_fit = f[idx]
                else:
                    X_fit = X
                    f_fit = f
                
                # Fit RBF
                rbf = Rbf(X_fit, f_fit, function='gaussian', smooth=1e-3)
                
                # Generate candidate points
                n_candidates = 200
                candidates = []
                for _ in range(n_candidates):
                    x_cand = np.array([rng.uniform(-5.0, 5.0) for _ in range(dim)])
                    candidates.append(x_cand)
                candidates = np.array(candidates)
                
                # Predict values and compute EI
                f_pred = rbf(candidates)
                f_min_current = np.min(f)
                
                # Expected Improvement
                f_mean = np.mean(f)
                f_std = np.std(f)
                
                # Simple EI: maximize (f_min_current - f_pred) / (f_std + 1e-8)
                # This encourages exploration where prediction is uncertain
                ei = (f_min_current - f_pred) / (np.abs(f_pred - f_mean) / (f_std + 1e-8) + 1.0)
                
                # Select best candidate
                idx_best = np.argmax(ei)
                x_new = candidates[idx_best]
                
            except Exception:
                # Fallback to random point
                x_new = np.array([rng.uniform(-5.0, 5.0) for _ in range(dim)])
            
            evaluate(x_new)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
