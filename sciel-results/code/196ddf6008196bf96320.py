import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration - sample many points to find good basins
        # Schaffers F7 is multimodal, so we need to explore broadly first
        # Use a stratified sampling approach in [-5, 5]^5
        
        # First, try the origin and a few deterministic points
        deterministic_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
        ]
        
        for p in deterministic_points:
            if calls >= budget or time_left() < 5.0:
                break
            evaluate(p)
        
        # Random sampling with Halton-like quasi-random sequence for better coverage
        # Use simple LCG-based quasi-random for reproducibility and speed
        n_global = min(80, budget - calls - 100)  # Leave budget for local refinement
        for i in range(n_global):
            if calls >= budget or time_left() < 5.0:
                break
            # Use a simple quasi-random sequence
            # Sobol-like: use bit-reversal for each dimension
            x = np.zeros(dim)
            for d in range(dim):
                # Generate quasi-random number in [0, 1)
                val = ((i + 1) * (d + 1) * 1103515245 + 12345) & 0x7FFFFFFF
                val = val / 0x7FFFFFFF
                x[d] = -5.0 + 10.0 * val
            evaluate(x)
        
        # Also try some random points
        n_random = min(40, budget - calls - 100)
        for _ in range(n_random):
            if calls >= budget or time_left() < 5.0:
                break
            x = np.random.uniform(-5, 5, dim)
            evaluate(x)
        
        # Phase 2: Local refinement using Nelder-Mead on the best points
        # Sort all evaluated points by fitness and refine the top few
        
        # We need to track all points, so let's keep a list
        # Actually, we only tracked the best. Let's do a more careful approach.
        # Reset and do it properly.
        
        # Let's restart with a cleaner approach that tracks all points
        
        # Actually, let me restructure: do global sampling, keep track of all points,
        # then refine the best ones with Nelder-Mead.
        
        # Since we already made some calls, let's just proceed with Nelder-Mead on best_x
        # and also try a few more starting points for local search.
        
        # Try Nelder-Mead from the best point found so far
        if best_x is not None and calls < budget - 50 and time_left() > 10.0:
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'initial_simplex': best_x + 0.1 * np.eye(dim)
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: If budget remains, do additional local search from perturbed best point
        remaining = budget - calls
        if remaining > 20 and time_left() > 10.0 and best_x is not None:
            # Try a few perturbations of the best point and refine
            n_perturb = min(5, remaining // 30)
            for _ in range(n_perturb):
                if calls >= budget - 20 or time_left() < 5.0:
                    break
                x_pert = best_x + 0.5 * np.random.standard_normal(dim)
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = evaluate(x_pert)
                if f_pert < best_f:
                    # Refine this new best
                    if calls < budget - 20 and time_left() > 5.0:
                        try:
                            from scipy.optimize import minimize
                            result = minimize(
                                func, x_pert, method='Nelder-Mead',
                                options={
                                    'maxiter': 200,
                                    'xatol': 1e-10,
                                    'fatol': 1e-12
                                }
                            )
                            if result.fun < best_f:
                                best_f = result.fun
                                best_x = result.x.copy()
                        except Exception:
                            pass
        
        # Phase 4: Final polish with very fine Nelder-Mead if budget allows
        if best_x is not None and calls < budget - 10 and time_left() > 5.0:
            try:
                from scipy.optimize import minimize
                result = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-12,
                        'fatol': 1e-14
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
