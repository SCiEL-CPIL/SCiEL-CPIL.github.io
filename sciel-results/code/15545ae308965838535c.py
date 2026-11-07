"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + L-BFGS-B hybrid for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global exploration
        # Use about 50% of budget for CMA-ES
        cma_budget = int(budget * 0.50)
        
        try:
            from cma import CMA, CMAOptions
            import cma
            
            # Set up CMA-ES
            x0 = np.zeros(d)
            sigma0 = 2.0  # initial step size
            
            opts = {
                'maxiter': 200,
                'maxfevals': cma_budget,
                'bounds': [(-5, 5)] * d,
                'seed': 42,
                'verbose': -9,
                'tolfun': 1e-15,
                'tolx': 1e-12,
                'tolcond': 1e12,
                'popsize': max(4, int(4 + 3 * np.log(d))),
            }
            
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            while not es.stop() and calls < cma_budget and time_ok():
                solutions = es.ask()
                for sol in solutions:
                    if calls >= cma_budget:
                        break
                    fx = safe_func(sol)
                    es.tell(sol, fx)
                
                # Update best from CMA-ES
                if es.result.xbest is not None:
                    if es.result.fbest < best_fx:
                        best_fx = es.result.fbest
                        best_x = es.result.xbest.copy()
                        
        except Exception:
            # Fallback to random search if CMA-ES fails
            for _ in range(cma_budget):
                if not time_ok() or calls >= cma_budget:
                    break
                x = rng.uniform(-4, 4, d)
                safe_func(x)

        # Phase 2: L-BFGS-B from best point found by CMA-ES
        if time_ok() and best_x is not None and calls < budget - 20:
            try:
                from scipy.optimize import minimize
                
                lbfgs_budget = int((budget - calls) * 0.4)
                
                def lbfgs_func(v):
                    nonlocal calls
                    if calls >= budget:
                        return best_fx
                    return safe_func(v)
                
                res = minimize(
                    lbfgs_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': 100, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish
        if time_ok() and best_x is not None and calls < budget - 10:
            try:
                from scipy.optimize import minimize
                
                nm_calls = [0]
                
                def nm_func(v):
                    nonlocal calls
                    if nm_calls[0] >= budget - calls:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(v, -5, 5))
                
                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                    )
                    calls += nm_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        # Phase 4: Final L-BFGS-B restarts from perturbed points
        if time_ok() and best_x is not None and calls < budget - 30:
            try:
                from scipy.optimize import minimize
                
                for _ in range(3):
                    if not time_ok() or calls >= budget - 10:
                        break
                    
                    perturbation = rng.normal(0, 0.3, d)
                    x0 = np.clip(best_x + perturbation, -5, 5)
                    
                    def lbfgs_func2(v):
                        nonlocal calls
                        if calls >= budget:
                            return best_fx
                        return safe_func(v)
                    
                    try:
                        res = minimize(
                            lbfgs_func2,
                            x0,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * d,
                            options={'maxiter': 50, 'maxls': 30, 'ftol': 1e-15, 'gtol': 1e-12}
                        )
                        if res.fun < best_fx:
                            best_fx = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass
            except Exception:
                pass

        # Final Nelder-Mead polish if we have budget left
        if time_ok() and best_x is not None and calls < budget - 10:
            try:
                from scipy.optimize import minimize
                
                nm_calls = [0]
                
                def nm_func2(v):
                    nonlocal calls
                    if nm_calls[0] >= budget - calls:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(v, -5, 5))
                
                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        nm_func2,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                    )
                    calls += nm_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
