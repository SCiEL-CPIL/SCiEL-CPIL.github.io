"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol quasi-random global exploration followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        # Sobol quasi-random sequence for global exploration
        def halton_sequence(n, dim, base=None):
            """Generate Halton sequence points in [0,1]^dim."""
            if base is None:
                # First dim primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                base = primes[:dim]
            
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(dim):
                    p = base[j]
                    f = 1.0
                    i_val = i
                    while i_val > 0:
                        f /= p
                        i_val, r = divmod(i_val, p)
                        f *= r
                    point.append(f)
                points.append(point)
            return np.array(points)

        # Phase 1: Global exploration with Sobol/Halton points
        n_global = min(100, budget // 5)
        sobol_points = halton_sequence(n_global, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        sobol_points = sobol_points * 10.0 - 5.0

        best_x = None
        best_f = float('inf')
        local_starts = []

        for pt in sobol_points:
            if calls >= budget or time.time() > deadline:
                break
            f_val = evaluate(pt)
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
            # Keep top candidates for local refinement
            local_starts.append((f_val, pt.copy()))

        # Sort by function value and keep top candidates
        local_starts.sort(key=lambda x: x[0])
        n_local = min(10, len(local_starts), max(1, (budget - calls) // 40))

        # Phase 2: Local refinement with Nelder-Mead on top candidates
        for f_init, x_init in local_starts[:n_local]:
            if calls >= budget or time.time() > deadline:
                break
            
            # Use a callback to track calls
            def nm_callback(xk):
                nonlocal calls
                return calls >= budget or time.time() > deadline
            
            try:
                result = minimize(
                    lambda x: evaluate(x),
                    x_init,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True},
                    callback=nm_callback
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        # Phase 3: If budget remains, do additional random restarts with NM
        if calls < budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining = budget - calls
            n_extra = min(5, remaining // 50)
            
            for _ in range(n_extra):
                if calls >= budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-5, 5, dim)
                f_rand = evaluate(x_rand)
                if f_rand < best_f:
                    best_f = f_rand
                    best_x = x_rand.copy()
                
                if calls < budget and time.time() < deadline:
                    def nm_callback2(xk):
                        nonlocal calls
                        return calls >= budget or time.time() > deadline
                    try:
                        result = minimize(
                            lambda x: evaluate(x),
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
                            callback=nm_callback2
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
