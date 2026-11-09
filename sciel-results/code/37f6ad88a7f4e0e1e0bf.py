"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES-like optimizer for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285  # safety margin before 300s limit

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls >= budget:
                return np.inf
            f = func(x)
            if not np.isfinite(f):
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start exploration
        rng = np.random.default_rng(42)
        
        # Try multiple starting points with different strategies
        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Random points
        for _ in range(15):
            start_points.append(rng.uniform(-5, 5, size=dim))
        
        # Points near typical Rosenbrock minimum (1,1,...,1)
        for _ in range(5):
            start_points.append(np.ones(dim) + rng.uniform(-1, 1, size=dim))
        
        # Points along the valley
        for _ in range(5):
            x = rng.uniform(-3, 3, size=dim)
            x[0] = 1.0  # Fix first coordinate near 1
            start_points.append(x)
        
        # Evaluate starting points
        evaluated_starts = []
        for x0 in start_points:
            if calls >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = safe_call(x0)
            if f0 < np.inf:
                evaluated_starts.append((f0, x0.copy()))
        
        if not evaluated_starts:
            return best_x, best_f
        
        # Sort by function value
        evaluated_starts.sort(key=lambda t: t[0])
        
        # Phase 2: Local optimization from best starting points
        # Use Nelder-Mead for local refinement
        n_local_starts = min(5, len(evaluated_starts))
        
        for i in range(n_local_starts):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = evaluated_starts[i][1]
            
            # Use scipy's Nelder-Mead with our safe_call wrapper
            try:
                result = minimize(
                    safe_call,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(100, budget - calls),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                continue
            
            if calls >= budget or time.time() > deadline:
                break
        
        # Phase 3: Fine-tuning with small random perturbations around best
        if calls < budget and time.time() < deadline:
            sigma = 0.1
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                x_try = best_x + sigma * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = safe_call(x_try)
                if f_try < best_f:
                    sigma *= 1.5
                else:
                    sigma *= 0.7
                    if sigma < 1e-6:
                        break
        
        return best_x, best_f
