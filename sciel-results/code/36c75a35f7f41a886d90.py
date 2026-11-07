import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_lbfgsb(x0, max_iter, gtol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'gtol': gtol,
                        'maxls': 50
                    }
                )
            except Exception:
                pass

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        def run_powell(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Powell',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol
                    }
                )
            except Exception:
                pass

        # Phase 1: Diverse Exploration
        # Use 20 starting points
        num_starts = 20
        sobol_sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_points = sobol_sampler.random(num_starts)
        sobol_points = sobol_points * 9.6 - 4.8 # Map to [-4.8, 4.8]
        
        starts = [np.zeros(n)]
        for p in sobol_points[:num_starts-1]:
            starts.append(p)

        # Allocation: 
        # Total budget 500.
        # Exploration: 20 starts.
        # Per start: ~20 calls for local search.
        # Reserve 100 calls for final polish.
        # So 400 calls for exploration. 400/20 = 20 calls per start.
        
        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            
            # 1. L-BFGS-B: Fast convergence
            # Allow up to 10 iterations (calls depend on step, but maxiter limits iterations)
            # L-BFGS-B usually uses 1 call per iteration + line search calls.
            # Let's be conservative with max_iter.
            run_lbfgsb(x0, max_iter=10, gtol=1e-8)
            
            # 2. Powell: Robust polishing
            # Start from the best point found so far in this local region? 
            # Or just x0? Using best_x is better if it's close.
            # But best_x might be from a previous start.
            # Let's use the point that L-BFGS-B converged to. 
            # We don't have direct access to the result of minimize, but best_x is updated.
            # If best_x changed significantly, it's likely in this basin.
            # Let's just run Powell from x0? No, from the current best_x is risky if it's far.
            # Let's run Powell from x0? No, L-BFGS-B already moved it.
            # We need the result of L-BFGS-B. 
            # We can modify run_lbfgsb to return the result.
            
            # Let's refactor to return results.
            pass

        # Let's restart with a cleaner structure that returns results.

        return best_x, best_f
