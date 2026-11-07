import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        evaluated_points = []
        
        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            # Ensure x is a valid numpy array of correct shape
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(dim)
            val = func(x)
            calls_made += 1
            evaluated_points.append((x.copy(), val))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(12345)
        
        # Phase 1: Space-filling sampling using Halton sequence
        # This provides better coverage of the domain than pure random or structured points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_phase1 = min(60, budget)
        phase1_points = []
        
        # Generate Halton sequence points in the box [-5, 5]^dim
        for i in range(1, n_phase1 + 1):
            point = []
            for d in range(dim):
                # Use different primes for each dimension
                base = [2, 3, 5, 7, 11][d]
                val = halton(i, base)
                # Map to [-5, 5]
                point.append(-5 + 10 * val)
            phase1_points.append(np.array(point))
        
        # Also add some points near known Katsuura structures
        # Polynomial part minimum near x=1
        for i in range(10):
            x = np.ones(dim) + rng.normal(0, 0.3, size=dim)
            phase1_points.append(x)
        
        # Points near cosine minima
        for i in range(10):
            x = rng.uniform(-5, 5, size=dim)
            x = np.round(x / np.pi) * np.pi + rng.normal(0, 0.05, size=dim)
            phase1_points.append(x)
        
        # Shuffle to mix types
        np.random.shuffle(phase1_points)
        
        for x in phase1_points[:n_phase1]:
            safe_call(x)
            if calls_made >= budget or time.time() > deadline:
                break
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Multi-start CMA-ES on top candidates
        evaluated_points.sort(key=lambda item: item[1])
        
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
        
        top_k = min(8, len(evaluated_points))
        
        for i in range(top_k):
            if calls_made >= budget or time.time() > deadline:
                break
            
            x0 = evaluated_points[i][0].copy()
            
            # Allocate budget for this run
            runs_remaining = top_k - i
            maxfev = min(100, max(30, remaining // runs_remaining))
            if maxfev < 20:
                break
            
            def cma_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return best_f
                return safe_call(x)
            
            try:
                res = minimize(
                    cma_func,
                    x0,
                    method='CMA',
                    options={
                        'maxiter': 50,
                        'maxfev': maxfev,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'bounds': [(-5, 5)] * dim,
                        'seed': 42 + i
                    }
                )
            except Exception:
                pass
        
        # Phase 3: If we still have budget, do a final local refinement with Nelder-Mead
        remaining = budget - calls_made
        if remaining > 20 and best_x is not None:
            def nm_func(x):
                if calls_made >= budget or time.time() > deadline:
                    return best_f
                return safe_call(x)
            
            try:
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'maxfev': min(remaining, 50),
                        'xatol': 1e-6,
                        'fatol': 1e-6
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f
