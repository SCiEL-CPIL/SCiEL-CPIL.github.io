"""Combined optimizer: multi-start global sampling + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    
    Strategy:
    1. Phase 1: Global exploration using a mix of random and quasi-random points.
       This quickly identifies promising regions in the shifted/rotated landscape.
    2. Phase 2: Local refinement using Nelder-Mead simplex method starting from
       the best points found in Phase 1.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def _count(self):
        self.calls += 1
        return self.calls

    def _evaluate(self, func, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        self._count()
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def _sobol_like(self, n, dim, rng):
        """Generate quasi-random points using a simple Halton-like sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Use a prime base for each dimension
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = bases[d % len(bases)]
            for i in range(n):
                # Halton sequence value
                f = 1.0
                r = 0.0
                idx = i + 1
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                points[i, d] = r
        # Map to [-5, 5]
        points *= 10.0 - 10.0
        points += -5.0
        return points

    def __call__(self, func):
        self.deadline = time.time() + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Global exploration
        # Use a mix of random and quasi-random points
        n_global = 150
        points = np.zeros((n_global, self.dim))
        
        # First 100 points: quasi-random (Halton-like)
        quasi_points = self._sobol_like(100, self.dim, rng)
        points[:100] = quasi_points
        
        # Next 50 points: random
        points[100:] = rng.uniform(-5, 5, (50, self.dim))
        
        # Evaluate global points
        for i in range(n_global):
            if time.time() > self.deadline:
                break
            self._evaluate(func, points[i])
        
        # Phase 2: Local refinement from top candidates
        # We need to track which points were evaluated and their values
        # Re-evaluate isn't needed, we just use the best few from Phase 1
        
        # Get top 5 starting points for local optimization
        # Since we didn't store all, let's re-evaluate a smaller set or use best_x
        # Actually, let's do a few more random restarts with local polish
        
        n_local = 5
        remaining_budget = self.budget - self.calls
        
        for k in range(n_local):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            # Start from a perturbation of the current best or a new random point
            if k == 0:
                x0 = self.best_x.copy()
            else:
                x0 = self.best_x + rng.normal(0, 0.5, self.dim)
                x0 = np.clip(x0, -5, 5)
            
            # Use Nelder-Mead for local optimization
            # We need a wrapper that counts calls
            def objective(x):
                if self.calls >= self.budget or time.time() > self.deadline:
                    return self.best_f  # Return best known to stop early
                return self._evaluate(func, x)
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass
        
        # Final check: if we have budget left, do a few more random perturbations
        while self.calls < self.budget and time.time() < self.deadline:
            x = self.best_x + rng.normal(0, 0.1, self.dim)
            x = np.clip(x, -5, 5)
            self._evaluate(func, x)
        
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
