"""Rewritten optimizer for BBOB-style Rosenbrock f8 with SLSQP + ES strategy."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid optimizer: Random exploration + SLSQP local refinement + (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            
            x_clip = np.clip(x, -5.0, 5.0)
            
            try:
                val = func(x_clip)
                val = float(val)
                if np.isfinite(val):
                    calls_made += 1
                    if val < best_f:
                        best_f = val
                        best_x = x_clip.copy()
                    return val
                else:
                    calls_made += 1
                    return 1e10
            except Exception:
                calls_made += 1
                return 1e10

        # Phase 1: Random exploration
        n_explore = 40
        seeds = []
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_func(x)
            seeds.append((x.copy(), fx))

        # Sort seeds by function value
        seeds.sort(key=lambda s: s[1])
        
        # Select diverse seeds for local optimization
        max_seeds = 6
        selected_seeds = []
        for s in seeds:
            if len(selected_seeds) >= max_seeds:
                break
            diverse = True
            for ts in selected_seeds:
                if np.linalg.norm(s[0] - ts[0]) < 1.2:
                    diverse = False
                    break
            if diverse:
                selected_seeds.append(s)
        
        # Ensure the absolute best is always included
        if seeds and selected_seeds:
            if selected_seeds[0][1] != seeds[0][1]:
                selected_seeds[0] = seeds[0]

        # Phase 2: SLSQP local optimization from selected seeds
        for i, (sx, sf) in enumerate(selected_seeds):
            if not time_ok() or calls_made >= budget - 15:
                break
            
            remaining = budget - calls_made
            if i == 0:
                maxiter = 80
            else:
                maxiter = max(15, remaining // (len(selected_seeds) - i + 1))
            
            try:
                result = minimize(
                    safe_func, 
                    sx, 
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-12,
                        'eps': 1e-6
                    }
                )
            except Exception:
                pass

        # Phase 3: Polish with (1+1)-ES
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            
            while calls_made < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.05, 3.0)
                else:
                    sigma = max(sigma * 0.9, 1e-6)
                    
                if sigma < 1e-5:
                    break

        return best_x, best_f
