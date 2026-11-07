"""Rewritten program for BBOB-style Rastrigin optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Hybrid Differential Evolution (Global) + CMA-ES (Local Polish) for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration with Differential Evolution
        # Allocate ~65% of budget to DE
        de_budget = int(budget * 0.65)
        
        # DE is robust to multimodality and rotation
        try:
            res_de = differential_evolution(
                counted_func,
                bounds=[(-5, 5)] * dim,
                strategy='best1bin',
                maxiter=200,
                maxfev=de_budget,
                popsize=15,
                tol=1e-12,
                seed=42,
                polish=False,  # Disable internal polish to save calls
                init='latinhypercube'
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            pass

        # Phase 2: Local Polish with CMA-ES
        # CMA-ES adapts to rotation, crucial for the final convergence
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 5:
            try:
                res_cma = minimize(
                    counted_func,
                    best_x,
                    method='CMA-ES',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining_budget,
                        'tol': 1e-12,
                        'seed': 123,
                        'popsize': 20,
                        'beta1': 0.5,
                        'beta2': 0.5
                    }
                )
                if res_cma.fun < best_f:
                    best_f = res_cma.fun
                    best_x = res_cma.x.copy()
            except Exception:
                # Fallback to Nelder-Mead if CMA-ES fails or is unavailable
                try:
                    res_nm = minimize(
                        counted_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
