"""Multi-stage Powell-based optimizer for BBOB f19 (Griewank-Rosenbrock).

Structural change: replace CMA-ES with Powell's method for local search.
Powell is more robust for the Griewank-Rosenbrock landscape because it
directly handles the narrow valleys without requiring covariance adaptation.
Uses diverse initializations and a final Nelder-Mead polish.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-stage Powell optimizer with diverse initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _powell_local(self, x0, f0, rng, budget, deadline):
        """Run Powell's method for at most `budget` evaluations.
        Returns (best_x, best_f, n_calls)."""
        n = self.dim
        if budget < 2:
            return x0, f0, 0

        calls = 0
        best_x = np.array(x0, dtype=float).copy()
        best_f = f0

        def fobj(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = float(self._f(x))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            res = minimize(fobj, x0, method='Powell',
                           bounds=[(-5.0, 5.0)] * n,
                           options={'maxiter': 50, 'maxfev': max(1, budget),
                                    'xtol': 1e-8, 'ftol': 1e-10})
        except Exception:
            pass

        return best_x, best_f, calls

    def _nelder_polish(self, x0, f0, deadline):
        """Final Nelder-Mead polish. Returns (best_x, best_f, n_calls)."""
        calls = 0
        best_x = np.array(x0, dtype=float).copy()
        best_f = f0

        def fobj(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = float(self._f(x))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            # Create initial simplex
            x0 = np.array(x0, dtype=float)
            delta = 0.1
            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            res = minimize(fobj, x0, method='Nelder-Mead',
                           options={'maxiter': 100, 'maxfev': 100,
                                    'xatol': 1e-8, 'fatol': 1e-10,
                                    'initial_simplex': np.array(simplex)})
        except Exception:
            pass

        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(12345)
        self._f = func

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Generate diverse initial points
        # Use structured + random initialization
        init_points = []
        
        # Origin
        init_points.append(np.zeros(self.dim))
        
        # Axis-aligned points (helps with Griewank structure)
        for axis in range(self.dim):
            for sign in [-1, 1]:
                pt = np.zeros(self.dim)
                pt[axis] = sign * 2.0
                init_points.append(pt)
        
        # Random points in the box
        n_random = 10
        for _ in range(n_random):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))

        # Phase 2: Evaluate all initial points and keep top candidates
        evaluated = []
        for x_init in init_points:
            if total_calls >= self.budget or time.time() >= deadline - 50:
                break
            x_init = np.clip(x_init, -5.0, 5.0)
            f_init = float(func(x_init))
            total_calls += 1
            evaluated.append((f_init, x_init.copy()))
            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()

        if best_x is None:
            x0 = np.zeros(self.dim)
            best_f = float(func(x0))
            total_calls += 1
            best_x = x0.copy()

        # Sort by fitness and take top candidates for local search
        evaluated.sort(key=lambda x: x[0])
        top_k = min(5, len(evaluated))
        
        # Phase 3: Powell local search on top candidates
        remaining_budget = self.budget - total_calls
        budget_per_local = max(10, remaining_budget // max(1, top_k))
        
        for i in range(top_k):
            if total_calls >= self.budget or time.time() >= deadline - 30:
                break
            
            f_init, x_init = evaluated[i]
            alloc = min(budget_per_local, self.budget - total_calls)
            if alloc < 5:
                break

            x_best, f_best, n_calls = self._powell_local(
                x_init, f_init, rng, alloc, deadline)
            total_calls += n_calls
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 4: Nelder-Mead polish on the best point
        remaining = self.budget - total_calls
        if remaining >= 10 and time.time() < deadline - 10:
            x_pol, f_pol, n_pol = self._nelder_polish(best_x, best_f, deadline)
            total_calls += n_pol
            if f_pol < best_f:
                best_f = f_pol
                best_x = x_pol.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
