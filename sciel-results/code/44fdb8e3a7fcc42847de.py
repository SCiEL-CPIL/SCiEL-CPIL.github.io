"""Adaptive multi-start Nelder-Mead + differential evolution for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random


class Optimizer:
    """Multi-start Nelder-Mead + DE optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        all_x = []
        all_f = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f, all_x, all_f
            if eval_count >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            f_val = float(func(x))
            all_x.append(x.copy())
            all_f.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        rng = np.random.default_rng(12345)

        # --- Phase 1: Sobol initialization ---
        from scipy.stats import qmc
        n_init = 40
        sobol = qmc.Sobol(d=dim, seed=42)
        points = sobol.random(n_init)
        for i in range(n_init):
            if eval_count >= budget - 100 or time.time() > deadline:
                break
            pt = points[i] * 10.0 - 5.0
            wrapped_func(pt)

        if eval_count < budget - 100 and time.time() <= deadline:
            wrapped_func(np.zeros(dim))

        # --- Phase 2: Differential Evolution ---
        def differential_evolution(x_best, n_gen, pop_size):
            nonlocal eval_count, best_x, best_f
            pop = [x_best + rng.normal(0, 1.5, dim) for _ in range(pop_size)]
            pop = [np.clip(p, -5.0, 5.0) for p in pop]
            f_pop = [wrapped_func(p) for p in pop]
            best_idx = np.argmin(f_pop)
            best_val = f_pop[best_idx]
            best_pt = pop[best_idx].copy()

            for gen in range(n_gen):
                if eval_count >= budget - 50 or time.time() > deadline:
                    break
                for i in range(pop_size):
                    if eval_count >= budget - 50 or time.time() > deadline:
                        break
                    # Select three distinct individuals
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    if a == i or b == i or c == i:
                        continue
                    F = 0.8 + rng.random() * 0.2
                    v = pop[a] + F * (pop[b] - pop[c])
                    v = np.clip(v, -5.0, 5.0)
                    f_v = wrapped_func(v)
                    if f_v < f_pop[i]:
                        pop[i] = v
                        f_pop[i] = f_v
                        if f_v < best_val:
                            best_val = f_v
                            best_pt = v.copy()
                            if f_v < best_f:
                                best_f = f_v
                                best_x = v.copy()
            return best_pt, best_val

        if eval_count < budget - 50 and time.time() < deadline:
            de_x, de_f = differential_evolution(best_x, n_gen=8, pop_size=12)
            if de_f < best_f:
                best_f = de_f
                best_x = de_x.copy()

        # --- Phase 3: Multi-start Nelder-Mead ---
        def nelder_mead(x0, max_iter=50, tol=1e-8):
            nonlocal eval_count
            if eval_count >= budget - 10 or time.time() > deadline:
                return x0, wrapped_func(x0)
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': tol,
                        'fatol': 1e-8,
                        'initial_simplex': None,
                        'adaptive': True
                    }
                )
                return result.x, result.fun
            except Exception:
                return x0, wrapped_func(x0)

        # Generate diverse starting points
        candidates = [best_x.copy()]

        if all_x:
            idx_sorted = np.argsort(all_f)
            for idx in idx_sorted[:10]:
                pt = all_x[idx].copy()
                is_distinct = True
                for c in candidates:
                    if np.linalg.norm(pt - c) < 0.5:
                        is_distinct = False
                        break
                if is_distinct:
                    candidates.append(pt)
                    if len(candidates) >= 8:
                        break

        # Add random points around best
        for _ in range(5):
            if len(candidates) >= 12:
                break
            pt = best_x + rng.normal(0, 0.5, dim)
            pt = np.clip(pt, -5.0, 5.0)
            is_distinct = True
            for c in candidates:
                if np.linalg.norm(pt - c) < 0.3:
                    is_distinct = False
                    break
            if is_distinct:
                candidates.append(pt)

        for cand in candidates:
            if eval_count >= budget - 5 or time.time() > deadline:
                break
            nm_x, nm_f = nelder_mead(cand, max_iter=40)
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()

        # --- Phase 4: Final L-BFGS-B refinement ---
        if eval_count < budget - 5 and time.time() < deadline:
            bounds = [(-5.0, 5.0)] * dim
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 30, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # --- Phase 5: Final local random search ---
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.02, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
