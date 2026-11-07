"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + (1+1)-ES polish with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            fx = func(np.asarray(x, dtype=np.float64))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        # Phase 1: Multi-start L-BFGS-B
        # L-BFGS-B converges fast on smooth landscapes like Schwefel
        n_starts = 8
        start_points = []

        # Center
        start_points.append(np.zeros(self.dim))
        # Axis-aligned points at various magnitudes
        for i in range(self.dim):
            p = np.zeros(self.dim)
            p[i] = 3.0
            start_points.append(p)
        # Additional random points
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, self.dim)
            if all(np.linalg.norm(p - q) > 1.5 for q in start_points):
                start_points.append(p)

        # Allocate budget: ~70% to L-BFGS-B, ~30% to ES polish
        lbfgs_budget = int(self.budget * 0.7)
        per_start_budget = max(15, lbfgs_budget // n_starts)

        bounds = [(-5.0, 5.0)] * self.dim

        for sp in start_points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            lbfgs_calls_this = [0]

            def lbfgs_obj(x):
                nonlocal lbfgs_calls_this
                if lbfgs_calls_this[0] >= per_start_budget:
                    return best_f
                lbfgs_calls_this[0] += 1
                return eval_func(x)

            try:
                res = minimize(
                    lbfgs_obj,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': per_start_budget,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                    }
                )
            except Exception:
                pass

        # Phase 2: Nelder-Mead simplex refinement from best point
        if best_x is not None and call_count < self.budget:
            nm_calls = [0]
            nm_budget = self.budget - call_count

            def nm_obj(x):
                if nm_calls[0] >= nm_budget:
                    return best_f
                nm_calls[0] += 1
                return eval_func(np.clip(x, -5, 5))

            # Multiple Nelder-Mead runs with diverse initial simplex offsets
            nm_starts = 4
            for k in range(nm_starts):
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Create diverse initial simplex around best_x
                scale = 0.5 if k == 0 else 1.0
                x0 = best_x.copy()
                if k > 0:
                    x0 = np.clip(x0 + scale * rng.standard_normal(self.dim), -5, 5)

                # Initialize simplex with diverse vertices
                simplex = [x0.copy()]
                for i in range(self.dim):
                    v = x0.copy()
                    v[i] += scale * (1.0 if i == 0 else 0.5)
                    simplex.append(np.clip(v, -5, 5))

                nm_calls[0] = 0
                per_nm_budget = max(10, (nm_budget - call_count) // (nm_starts - k))

                def nm_obj_limited(x):
                    if nm_calls[0] >= per_nm_budget:
                        return best_f
                    nm_calls[0] += 1
                    return eval_func(np.clip(x, -5, 5))

                try:
                    res = minimize(
                        nm_obj_limited,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': per_nm_budget,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 3: Final fine-tuning Nelder-Mead with tight tolerances
        if best_x is not None and call_count < self.budget:
            remaining = self.budget - call_count
            if remaining >= 5:
                final_calls = [0]

                def final_nm_obj(x):
                    if final_calls[0] >= remaining:
                        return best_f
                    final_calls[0] += 1
                    return eval_func(np.clip(x, -5, 5))

                try:
                    res = minimize(
                        final_nm_obj,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
