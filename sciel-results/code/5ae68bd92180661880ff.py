"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Strategy: Global exploration with Sobol sampling, followed by Powell's Method
(Derivative-Free Direct Search) for local refinement. Powell's method is
robust to the multimodality of the Rastrigin component and adapts search
directions without relying on gradient estimates.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        # Best solution tracking
        best_x = None
        best_f = np.inf
        
        def eval_func(x):
            nonlocal best_x, best_f
            # Clip to box
            x_clip = np.clip(x, -5.0, 5.0)
            fx = func(x_clip)
            if fx < best_f:
                best_f = fx
                best_x = x_clip.copy()
            return fx

        # Phase 1: Global exploration
        # Use Sobol sequence for better space-filling properties than uniform random
        n_initial = 60
        if n_initial >= budget:
            n_initial = budget // 2

        # Generate Sobol points in [0, 1]^d
        sampler = qmc.Sobol(d=d, scramble=True, seed=42)
        sobol_samples = sampler.random(n_initial)
        
        # Map to [-5, 5]^d
        points = sobol_samples * 10.0 - 5.0
        
        # Ensure we have at least the origin or a few key points
        # If n_initial is small, we might miss the origin. Let's force origin if not present.
        # Actually, Sobol is random-ish. Let's just evaluate them.
        
        initial_results = []
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = points[i]
            fx = eval_func(x)
            initial_results.append((fx, x))
            
            if i + 1 >= budget:
                break
                
        # Add origin explicitly if not evaluated (Sobol might not hit 0)
        # Check if any point is close to 0
        origin_evaluated = False
        for fx, x in initial_results:
            if np.allclose(x, 0, atol=1e-6):
                origin_evaluated = True
                break
        
        if not origin_evaluated and len(initial_results) < budget:
             x0 = np.zeros(d)
             fx0 = eval_func(x0)
             initial_results.append((fx0, x0))

        if len(initial_results) == 0:
            x0 = np.zeros(d)
            eval_func(x0)
            return best_x, best_f

        # Sort initial results
        initial_results.sort(key=lambda item: item[0])
        
        # Select top candidates for local refinement
        top_k = min(4, len(initial_results))
        candidates = []
        seen = set()
        for fx, x in initial_results:
            key = tuple(np.round(x, 2))
            if key not in seen:
                seen.add(key)
                candidates.append((fx, x))
            if len(candidates) >= top_k:
                break
        
        # Phase 2: Local Refinement with Powell's Method
        calls_made = len(initial_results)
        remaining_calls = budget - calls_made
        
        if remaining_calls <= 0:
            return best_x, best_f
            
        for i, (fx_start, x_start) in enumerate(candidates):
            if time.time() > deadline:
                break
            if remaining_calls <= 10:
                break
                
            # Powell's method doesn't have a direct 'maxiter' that maps 1:1 to function calls
            # like L-BFGS. We must wrap it to stop when budget is exhausted.
            
            local_calls = [0]
            def local_eval(x):
                local_calls[0] += 1
                if local_calls[0] > remaining_calls:
                    raise ValueError("Budget exceeded")
                return eval_func(x)
            
            try:
                # Powell's method is a direct search method.
                # It adapts the simplex/axes.
                # ftol and xtol control convergence.
                result = minimize(local_eval, x_start, method='Powell',
                                  options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-10})
                
                remaining_calls -= local_calls[0]
                
            except Exception:
                # If budget exceeded or other error, we keep the best found so far
                pass
                
            if time.time() > deadline - 5:
                break
        
        return best_x, best_f
