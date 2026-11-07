import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Memetic Algorithm using Nelder-Mead for local exploitation and 
    adaptive random jumps for global exploration.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish by 300s.
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("time exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        rng = np.random.default_rng(42)

        # 1. Initialization: Diverse starts
        starts = [np.zeros(dim)]
        for i in range(dim):
            for val in (-4, -2, 0, 2, 4):
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        
        # Add random points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))

        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)

        for s in unique_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                update_best(s, f)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        current_x = best_x.copy()
        current_f = best_f
        step_size = 2.0  # Initial global jump size
        min_step = 0.01
        max_step = 10.0

        # 2. Memetic Loop: Global Jump + Local Refinement
        while calls[0] < budget and time.time() < deadline:
            # Generate a global candidate
            direction = rng.standard_normal(dim)
            perturbed_x = np.clip(current_x + step_size * direction, -5, 5)
            
            new_x, new_f = perturbed_x, np.inf
            
            # Local Refinement using Nelder-Mead
            # We limit maxiter to prevent wasting too much budget on one local minimum
            # If the function is smooth, NM converges fast. If not, it stops.
            try:
                res = minimize(
                    wrapped,
                    perturbed_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                new_x, new_f = res.x, res.fun
            except Exception:
                pass

            # If NM failed or didn't improve, try a quick L-BFGS-B if we suspect smoothness?
            # No, stick to NM for robustness against Rastrigin/Gallagher.
            
            if new_f < current_f:
                current_x = new_x.copy()
                current_f = new_f
                update_best(current_x, current_f)
                # Exploit: reduce step size
                step_size *= 0.7
                if step_size < min_step:
                    step_size = min_step
            else:
                # Explore: increase step size
                step_size *= 1.2
                if step_size > max_step:
                    step_size = max_step

        # 3. Final Polish
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            try:
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        return best_x, best_f
