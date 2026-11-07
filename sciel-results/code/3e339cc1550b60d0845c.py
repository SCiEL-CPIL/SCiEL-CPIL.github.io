"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid global scout + Powell optimizer with ES refinement for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track best solution and function value
        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time exceeded")
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Global Scout with strategic points to find a good basin
        # Rosenbrock has a valley; the minimum is near (1,...,1) in standard form,
        # but hidden shift/rotation moves it. Try multiple strategic starting points.
        rng = np.random.default_rng(42)

        # Candidate starting points: origin, nominal min, and strategic grid points
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            -np.ones(self.dim),
        ]
        
        # Add corners of the box (±5)
        for i in range(2**self.dim):
            p = np.zeros(self.dim)
            for j in range(self.dim):
                if (i >> j) & 1:
                    p[j] = 5.0
                else:
                    p[j] = -5.0
            starts.append(p)
        
        # Add mid-points (±2.5)
        for i in range(2**self.dim):
            p = np.zeros(self.dim)
            for j in range(self.dim):
                if (i >> j) & 1:
                    p[j] = 2.5
                else:
                    p[j] = -2.5
            starts.append(p)
        
        # Add some random starts spread across the domain
        for _ in range(5):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        # Evaluate all starting points
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            try:
                f_val = wrapped(s_clipped)
            except Exception:
                break

        # Phase 2: Run Powell's method from the best scout point
        if best_x is not None:
            # Powell's method with tight tolerances
            result = minimize(
                wrapped,
                best_x,
                method='Powell',
                options={
                    'maxiter': 300,
                    'maxfev': max(1, self.budget - call_count - 50),  # reserve some budget
                    'xtol': 1e-10,
                    'ftol': 1e-12,
                }
            )
            # Update best
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)

            # Phase 3: (1+1)-ES refinement to escape local minima and polish
            if call_count < self.budget and time.time() < deadline:
                x_es = best_x.copy()
                fx_es = best_f
                sigma = 0.1  # Start with small step for refinement
                max_es_calls = min(80, self.budget - call_count)
                
                for _ in range(max_es_calls):
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped(y)
                    if fy < fx_es:
                        x_es, fx_es = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                    
                    if fx_es < best_f:
                        best_f = fx_es
                        best_x = x_es.copy()

            # Phase 4: Final polish with Nelder-Mead from the very best
            if call_count < self.budget and time.time() < deadline:
                result2 = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'maxfev': max(1, self.budget - call_count - 20),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                    }
                )
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = np.clip(result2.x, -5.0, 5.0)

            # Phase 5: Another round of ES for final polish
            if call_count < self.budget and time.time() < deadline:
                x_es = best_x.copy()
                fx_es = best_f
                sigma = 0.05  # Even smaller step for final polish
                max_es_calls = min(50, self.budget - call_count)
                
                for _ in range(max_es_calls):
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped(y)
                    if fy < fx_es:
                        x_es, fx_es = y, fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    
                    if fx_es < best_f:
                        best_f = fx_es
                        best_x = x_es.copy()

        # Ensure we have a valid return
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < self.budget:
                best_f = func(best_x)
                call_count += 1
            else:
                best_f = 0.0  # fallback

        return best_x, best_f
# EVOLVE-BLOCK-END
