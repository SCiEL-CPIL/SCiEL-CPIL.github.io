"""Combined optimizer: multi-start global sampling + Nelder-Mead + Powell polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random


class Optimizer:
    """
    Multi-start Nelder-Mead + Powell optimizer for BBOB f17 (Schaffers F7).
    
    Strategy:
    1. Phase 1: Global exploration using random and quasi-random points (~25% of budget).
       This identifies promising regions in the shifted/rotated landscape.
    2. Phase 2: Local refinement using Nelder-Mead simplex method starting from
       the best points found in Phase 1.
    3. Phase 3: High-precision polish using Powell's method with tighter tolerances.
       This helps on flat regions where Nelder-Mead may stall.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Global exploration
        n_explore = max(20, int(self.budget * 0.25))
        
        # Initial random points
        for i in range(n_explore):
            if call_count >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)
        
        # Sobol-like quasi-random points for better coverage
        n_quasi = max(10, int(self.budget * 0.15))
        for i in range(n_quasi):
            if call_count >= self.budget or time.time() > deadline:
                break
            # Use Halton sequence approximation
            x = np.array([self._halton(i + 1, base) for base in [2, 3, 5, 7, 11][:self.dim]])
            x = x * 10 - 5  # Map to [-5, 5]
            evaluate(x)
        
        # Phase 2: Nelder-Mead local optimization from best points
        n_local = max(5, int(self.budget * 0.4))
        local_budget_each = max(10, n_local // 5)
        
        # Get top candidates for local optimization
        # We'll just use the best point found so far and a few nearby points
        if best_x is not None:
            candidates = [best_x.copy()]
            
            # Add some perturbed versions
            for _ in range(4):
                if call_count >= self.budget or time.time() > deadline:
                    break
                perturbation = rng.uniform(-1, 1, self.dim) * 0.5
                x_candidate = np.clip(best_x + perturbation, -5, 5)
                f_candidate = func(x_candidate)
                call_count += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = x_candidate.copy()
                candidates.append(x_candidate)
            
            # Run Nelder-Mead from each candidate
            for start_point in candidates:
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                # Limit function evaluations in this local search
                maxfev_local = min(local_budget_each, self.budget - call_count)
                if maxfev_local < 5:
                    break
                
                try:
                    result = minimize(
                        evaluate,
                        start_point,
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxfev_local,
                            'maxfev': maxfev_local,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )
                except Exception:
                    continue
                
                # Update best if improved
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
        
        # Phase 3: Powell polish for high precision
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and best_x is not None and time.time() < deadline:
            maxfev_powell = min(remaining_budget, 100)
            if maxfev_powell > 5:
                try:
                    result = minimize(
                        evaluate,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': maxfev_powell,
                            'maxfev': maxfev_powell,
                            'xatol': 1e-12,
                            'fatol': 1e-14
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        # Final check: if we have budget left, do one more very tight polish
        remaining_budget = self.budget - call_count
        if remaining_budget > 5 and best_x is not None and time.time() < deadline:
            maxfev_final = min(remaining_budget, 50)
            if maxfev_final > 3:
                try:
                    result = minimize(
                        evaluate,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': maxfev_final,
                            'maxfev': maxfev_final,
                            'xatol': 1e-15,
                            'fatol': 1e-16
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        return best_x, best_f
    
    def _halton(self, index, base):
        """Generate Halton sequence value."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
# EVOLVE-BLOCK-END
