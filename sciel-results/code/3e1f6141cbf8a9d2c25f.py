import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')
        self.dim = self.dim
        self.budget = self.budget
        rng = np.random.default_rng(42)
        
        def tracked_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            fx = float(func(x))
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x)
            return fx

        # 1. Initial Sampling
        # Use 20% of budget for initial random search
        n_samples = max(10, self.budget // 5)
        
        # Evaluate origin
        x0 = np.zeros(self.dim)
        try:
            f0 = tracked_func(x0)
        except Exception:
            return self.best_x, self.best_f

        samples = []
        for _ in range(n_samples - 1):
            x = rng.uniform(-5, 5, self.dim)
            try:
                f = tracked_func(x)
                samples.append((f, x))
            except Exception:
                break
        
        if not samples:
            return self.best_x, self.best_f

        # Sort by function value
        samples.sort(key=lambda t: t[0])
        
        # 2. Local Refinement
        remaining = self.budget - self.calls
        if remaining <= 0:
            return self.best_x, self.best_f

        # Refine top 5 candidates
        n_refine = min(5, len(samples))
        # Each Nelder-Mead iteration costs ~dim+1 calls
        # We want to use remaining budget
        calls_per_refine = remaining // max(1, n_refine)
        
        for i in range(n_refine):
            if self.calls >= self.budget:
                break
            
            f_init, x_init = samples[i]
            
            # Estimate max iterations for Nelder-Mead
            # Nelder-Mead does dim+1 function calls per iteration
            max_iter = max(1, calls_per_refine // (self.dim + 1))
            
            try:
                res = minimize(
                    tracked_func, 
                    x_init, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.array(res.x)
            except Exception:
                pass

        # 3. Final Polish / Random Restarts if budget left
        while self.calls < self.budget - 5:
            x = rng.uniform(-5, 5, self.dim)
            try:
                f = tracked_func(x)
            except Exception:
                break
            
            if f < self.best_f:
                # Quick refine
                calls_left = self.budget - self.calls
                if calls_left > self.dim + 1:
                    max_iter = max(1, calls_left // (self.dim + 1))
                    try:
                        res = minimize(
                            tracked_func, 
                            x, 
                            method='Nelder-Mead',
                            options={
                                'maxiter': max_iter, 
                                'xatol': 1e-8, 
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                        if res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = np.array(res.x)
                    except Exception:
                        pass

        return self.best_x, self.best_f
