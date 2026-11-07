"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid scout + Powell optimizer with ES refinement for Rosenbrock-style f8."""

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
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Scout with several starting points to find a good basin
        # Rosenbrock has a valley; the minimum is near (1,...,1) in standard form,
        # but hidden shift/rotation moves it. Try multiple starting points.
        rng = np.random.default_rng(42)

        # Candidate starting points: origin, nominal min, and random points
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim),
            np.array([1.0] * self.dim),
            np.array([0.5] * self.dim),
            np.array([-1.0] * self.dim),
        ]
        # Add more random starts spread across the domain for better valley coverage
        for _ in range(8):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))

        # Evaluate all starting points
        for s in starts:
            if call_count >= self.budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            f_val = wrapped(s_clipped)

        # Phase 2: Run Powell's method from the best scout point
        if best_x is not None:
            # Powell's method with tight tolerances
            result = minimize(
                wrapped,
                best_x,
                method='Powell',
                options={
                    'maxiter': 200,
                    'maxfev': max(1, self.budget - call_count),
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
                max_es_calls = min(100, self.budget - call_count)
                
                for _ in range(max_es_calls):
                    if call_count >= self.budget or time.time() > deadline:
                        break
                    y = np.clip(x_es + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = wrapped(y)
                    if fy < fx_es:
                        x_es, fx_es = y, fy
                        sigma *= 1.3
                    else:
                        sigma *= 0.85
                    
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
                        'maxiter': 100,
                        'maxfev': max(1, self.budget - call_count),
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
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                    
                    if fx_es < best_f:
                        best_f = fx_es
                        best_x = x_es.copy()

            # Phase 6: Final high-precision polish with Powell from the very best
            # Use very tight tolerances to squeeze out last-bit precision
            if call_count < self.budget and time.time() < deadline:
                remaining_fev = max(1, self.budget - call_count)
                result3 = minimize(
                    wrapped,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 150,
                        'maxfev': remaining_fev,
                        'xtol': 1e-14,
                        'ftol': 1e-16,
                    }
                )
                if result3.fun < best_f:
                    best_f = result3.fun
                    best_x = np.clip(result3.x, -5.0, 5.0)

        # Ensure we have a valid return
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
