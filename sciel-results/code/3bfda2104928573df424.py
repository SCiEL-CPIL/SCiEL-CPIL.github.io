"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube


class Optimizer:
    """Multistart Nelder-Mead optimizer with Latin Hypercube global initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        state = {
            'calls': 0,
            'best_x': None,
            'best_f': float('inf')
        }

        def wrapped_func(x):
            if state['calls'] >= total_budget:
                return state['best_f']
            state['calls'] += 1
            fx = func(x)
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x.copy()
            return fx

        # 1. Global Exploration using Latin Hypercube Sampling
        # Allocate 20% of budget to global search, between 10 and 50 calls
        n_global = max(10, min(50, int(total_budget * 0.2)))
        
        if total_budget <= n_global:
            sampler = LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=total_budget) * 10 - 5
            for p in points:
                wrapped_func(p)
            return state['best_x'], state['best_f']

        sampler = LatinHypercube(d=dim, seed=42)
        points = sampler.random(n=n_global) * 10 - 5
        
        fvals = []
        for p in points:
            f = wrapped_func(p)
            fvals.append(f)
        
        fvals = np.array(fvals)
        
        # Select top 3 points for local search
        top_k = 3
        if len(fvals) < top_k:
            top_k = len(fvals)
            
        sorted_indices = np.argsort(fvals)
        start_points = [points[i] for i in sorted_indices[:top_k]]

        # 2. Local Exploitation using Nelder-Mead
        # Run Nelder-Mead from each of the best starting points
        for x0 in start_points:
            if state['calls'] >= total_budget:
                break
            try:
                # Use a high maxiter, but the wrapper will stop it when budget is exhausted
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-5})
            except Exception:
                pass

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
