"""Rewritten optimizer: multi-start CMA-ES with L-BFGS-B polish for Lunacek f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import cma


class Optimizer:
    """Multi-start CMA-ES with L-BFGS-B polish for Lunacek f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        def f(x):
            if calls[0] >= budget:
                return 1e12
            if time.time() > self._deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)

        # ============================================================
        # Phase 1: Initial Sobol space-filling to find good basins
        # ============================================================
        try:
            from scipy.stats.qmc import Sobol
            n_sobol = min(64, budget - 50)
            if n_sobol > 10:
                sobol = Sobol(d=dim, scramble=True, seed=42)
                unit_samples = sobol.random(n_sobol)
                sobol_points = unit_samples * 10.0 - 5.0
                for x0 in sobol_points:
                    if calls[0] >= budget or time.time() > self._deadline:
                        break
                    fx0 = f(x0)
                    update_best(x0, fx0)
        except Exception:
            pass

        # ============================================================
        # Phase 2: Multi-start CMA-ES from best Sobol points
        # ============================================================
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            # Collect top candidate start points
            start_points = [best_x.copy()]
            
            # Add a few diverse start points
            rng = np.random.RandomState(42)
            for _ in range(3):
                x_rand = rng.uniform(-4, 4, size=dim)
                start_points.append(x_rand)
            
            # Add perturbations around best
            for scale in [1.0, 2.0]:
                for d in range(dim):
                    for sign in [1, -1]:
                        xp = best_x.copy()
                        xp[d] += sign * scale
                        xp = np.clip(xp, -5, 5)
                        start_points.append(xp)

            # Deduplicate and select top starts by initial evaluation
            evaluated_starts = []
            for sp in start_points:
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                fx_sp = f(sp)
                evaluated_starts.append((fx_sp, sp))
                update_best(sp, fx_sp)
            
            # Sort by function value and take top 3 distinct
            evaluated_starts.sort(key=lambda x: x[0])
            selected_starts = []
            for fx_s, x_s in evaluated_starts:
                if len(selected_starts) >= 3:
                    break
                if np.linalg.norm(x_s - selected_starts[0]) > 1.0 if selected_starts else True:
                    selected_starts.append(x_s)

            # Run CMA-ES from each selected start
            for i, x_start in enumerate(selected_starts):
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                
                remaining_calls = budget - calls[0]
                if remaining_calls < 30:
                    break
                
                # Allocate calls: more for first start
                max_cma_calls = min(remaining_calls - 20, max(30, int(remaining_calls * 0.4 / (len(selected_starts) - i))))
                
                try:
                    opts = {
                        'maxiter': 50,
                        'tolopt': 1e-10,
                        'tolfun': 1e-12,
                        'bounds': [(-5, 5)] * dim,
                        'verbose': -9,
                        'seed': 42 + i,
                        'CMA_stopping_criteria': [
                            'ifunc == 0', 'ixf == 0', 'ifac == 0', 'ifx == 0',
                            'igx == 0', 'ig == 0', 'igmax == 0', 'igmin == 0',
                            'igmaxmin == 0', 'ir == 0', 'is == 0', 'it == 0',
                            'ix == 0', 'ixf == 0', 'ifun == 0', 'if == 0'
                        ],
                        'CMA_popsize': max(4, dim + 2),
                        'CMA_sigma0': 1.5,
                        'CMA_cc': 0,
                        'CMA_cs': 0,
                        'CMA_mu': None,
                        'CMA_c1': None,
                        'CMA_cmup': None,
                    }
                    
                    es = cma.CMAEvolutionStrategy(x_start, 1.5, opts)
                    es._ask_evals = max_cma_calls
                    
                    while not es.stop() and calls[0] < budget and time.time() < self._deadline:
                        solutions = es.ask()
                        for sol in solutions:
                            if calls[0] >= budget or time.time() > self._deadline:
                                break
                            fx_sol = f(sol)
                            update_best(sol, fx_sol)
                        es.tell(solutions, [f(s) for s in solutions])
                    
                    if es.best['f'] < best_f:
                        best_f = es.best['f']
                        best_x = np.array(es.best['x'], dtype=float)
                        
                except Exception:
                    pass

        # ============================================================
        # Phase 3: L-BFGS-B polish from best CMA-ES result
        # ============================================================
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 50, 'ftol': 1e-15}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # ============================================================
        # Phase 4: Final Nelder-Mead refinement if budget remains
        # ============================================================
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': 50,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
