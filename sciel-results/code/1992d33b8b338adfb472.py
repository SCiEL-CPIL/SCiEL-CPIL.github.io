"""Multi-restart L-BFGS-B + Nelder-Mead optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """Multi-restart gradient-based optimization with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-restart L-BFGS-B from diverse starting points
        # Adaptive restart strategy: more restarts early, fewer later
        # Reserve budget for NM polish
        nm_budget = 150  # Reserve 150 calls for Nelder-Mead polish
        lbfgsb_budget = budget - nm_budget
        num_restarts = 10
        calls_per_restart = max(1, lbfgsb_budget // num_restarts)

        # Pre-evaluate x=0 to establish baseline early
        x0 = np.zeros(n)
        evaluate(x0)

        for r in range(num_restarts):
            if calls >= lbfgsb_budget or time.time() >= deadline:
                break
            
            # Generate diverse starting point
            if r == 0:
                # Already evaluated x=0, so start from a perturbed version
                x0 = rng.uniform(-1.0, 1.0, size=n)
            elif r == 1:
                # Try points along the Rosenbrock valley direction
                x0 = np.ones(n) * 1.0
            else:
                x0 = rng.uniform(-4.5, 4.5, size=n)
            
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Try L-BFGS-B from this starting point
            try:
                from scipy.optimize import minimize
                
                # Wrapper for L-BFGS-B
                def wrapped_lbfgsb(x):
                    return evaluate(x)
                
                result = minimize(
                    wrapped_lbfgsb,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': calls_per_restart,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )
                
            except Exception:
                pass

        # Phase 2: Nelder-Mead polish from the best point found
        # Use larger initial simplex and more aggressive convergence
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize
                
                x0 = best_x.copy()
                remaining = budget - calls
                
                def wrapped_nm(x):
                    return evaluate(x)
                
                # Nelder-Mead with larger initial simplex for better valley navigation
                initial_simplex = np.array([x0])
                for i in range(n):
                    perturbed = x0.copy()
                    perturbed[i] += 0.5  # Larger perturbation for Rosenbrock valley
                    initial_simplex = np.vstack([initial_simplex, perturbed])
                
                result = minimize(
                    wrapped_nm,
                    x0,
                    method='Nelder-Mead',
                    initial_simplex=initial_simplex,
                    options={
                        'maxiter': max(100, remaining),
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True,
                    }
                )
                
            except Exception:
                pass

        # Phase 3: Additional L-BFGS-B polish if budget remains
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from scipy.optimize import minimize
                
                x0 = best_x.copy()
                remaining = budget - calls
                
                def wrapped_lbfgsb_final(x):
                    return evaluate(x)
                
                result = minimize(
                    wrapped_lbfgsb_final,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max(50, remaining),
                        'ftol': 1e-16,
                        'gtol': 1e-12,
                    }
                )
                
            except Exception:
                pass

        return best_x, best_f
