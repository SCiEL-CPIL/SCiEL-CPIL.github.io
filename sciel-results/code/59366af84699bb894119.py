"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution optimizer for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_eval(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')

            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1

            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx if np.isfinite(fx) else float('inf')

        # Use differential evolution with careful parameter tuning
        # f22 is multimodal with 21 peaks, so DE is well-suited
        # We need to be careful about the budget: 500 calls total
        
        # DE parameters:
        # popsize: number of individuals in population
        # maxiter: maximum number of generations
        # Each generation uses popsize evaluations (approximately)
        # So total calls ≈ popsize * maxiter + initial population
        
        # Let's use a moderate population and fewer generations
        # popsize=20, maxiter=20 → ~400-450 calls
        # This leaves room for a final local refinement
        
        # Bounds
        bounds = [(-5.0, 5.0)] * dim
        
        # We'll run DE with a callback to track progress
        # DE uses 'differential_evolution' from scipy
        
        # Strategy: 
        # 1. Run DE with popsize=15, maxiter=25 → ~375 calls
        # 2. Use remaining budget for local Nelder-Mead refinement
        
        # Actually, let's be more careful. DE in scipy uses:
        # - Initial population of popsize individuals
        # - Each generation evaluates popsize new candidates
        # - Total evaluations ≈ popsize * (maxiter + 1)
        
        # With popsize=15, maxiter=25: 15*26 = 390 calls
        # Remaining: ~110 calls for refinement
        
        # But we need to handle the time limit carefully.
        # Let's use a smaller population to be safe.
        
        popsize = 12
        maxiter = 30  # 12 * 31 = 372 calls
        
        # Run differential evolution
        try:
            result = differential_evolution(
                safe_eval,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=False,  # We'll do our own refinement
                init='sobol',  # Better initial spread
                updating='immediate'
            )
        except Exception:
            pass

        # Local refinement with Nelder-Mead on the best point
        if calls_made < budget - 10 and time.time() < deadline:
            from scipy.optimize import minimize
            x0 = best_x.copy()
            try:
                res = minimize(
                    safe_eval,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we still have budget, do a few more random restarts + refine
        if calls_made < budget - 20 and time.time() < deadline:
            rng = np.random.default_rng(123)
            n_extra = min(5, (budget - calls_made) // 10)
            extra_results = []
            for _ in range(n_extra):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_eval(x)
                if np.isfinite(fx):
                    extra_results.append((fx, x.copy()))

            if extra_results:
                extra_results.sort(key=lambda t: t[0])
                fx0, x0 = extra_results[0]
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
                
                # Refine the best extra point
                if calls_made < budget - 5 and time.time() < deadline:
                    from scipy.optimize import minimize
                    try:
                        res = minimize(
                            safe_eval,
                            best_x.copy(),
                            method='Nelder-Mead',
                            options={
                                'maxiter': 50,
                                'xatol': 1e-10,
                                'fatol': 1e-10,
                                'adaptive': True
                            }
                        )
                        if np.isfinite(res.fun) and res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
