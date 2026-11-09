"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Two-phase optimizer: coarse random search + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return np.inf
            call_count += 1
            try:
                fx = float(func(x))
                if fx < best_f:
                    best_f = fx
                    best_x = np.array(x, dtype=float)
                return fx
            except Exception:
                return np.inf

        # Phase 1: Coarse random search to find basin of attraction
        # Use structured + random points, biased toward center since optimum is in [-4,4]^5
        rng = np.random.default_rng(42)
        
        # Center-biased initial points
        center_points = [np.zeros(d)]
        # Add axis-aligned points
        for i in range(d):
            for val in [-2.0, -1.0, 1.0, 2.0]:
                p = np.zeros(d)
                p[i] = val
                center_points.append(p)
        
        # Random points in [-4, 4]^5
        num_random = min(40, budget - len(center_points) - 50)
        for _ in range(max(0, num_random)):
            center_points.append(rng.uniform(-4, 4, d))
        
        # Evaluate all initial points
        for x in center_points:
            if call_count >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if best_x is None:
            best_x = np.zeros(d)
            wrapped_func(best_x)

        # Phase 2: CMA-ES refinement from the best point found
        remaining = budget - call_count
        if remaining > 20 and time.time() < deadline:
            try:
                from cma import CMA
                x0 = best_x.copy()
                sigma0 = 1.0  # Initial step size
                opts = {
                    'maxiter': max(1, remaining // 10),
                    'seed': 42,
                    'verbose': -9,  # Suppress output
                    'toloptbest': 1e-8,
                    'tolx': 1e-8,
                }
                
                # Use CMA-ES with our wrapped function
                es = CMA(x0, sigma0, {'bounds': [-5.0]*d, 'maxiter': max(1, remaining // 10), 'seed': 42, 'verbose': -9, 'toloptbest': 1e-8, 'tolx': 1e-8})
                
                while not es.stop() and call_count < budget and time.time() < deadline:
                    solutions = es.ask()
                    for sol in solutions:
                        if call_count >= budget or time.time() > deadline:
                            break
                        wrapped_func(sol)
                    es.tell(solutions, [wrapped_func(s) for s in solutions if call_count < budget])
                    
                if es.result.x is not None and len(es.result.x) == d:
                    if es.result.fbest < best_f:
                        best_f = es.result.fbest
                        best_x = np.array(es.result.x, dtype=float)
            except Exception:
                pass

        # Phase 3: If budget remains, do a final polish with Nelder-Mead
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
