"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


class Optimizer:
    """Hybrid differential evolution + L-BFGS-B for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 260  # safety margin within 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            call_count += 1
            if call_count > self.budget:
                return best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # --- Phase 1: Quick initial exploration to seed DE ---
        # Evaluate a few strategic points to get a rough sense of the landscape
        init_points = [np.zeros(self.dim)]
        for d in range(self.dim):
            for s in [-2.0, 2.0]:
                p = np.zeros(self.dim)
                p[d] = s
                init_points.append(p)
        for _ in range(10):
            init_points.append(rng.uniform(-3, 3, self.dim))

        for x in init_points:
            if call_count >= self.budget or time.time() > deadline - 60:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(self.dim), wrapped_func(np.zeros(self.dim))

        # --- Phase 2: Differential Evolution for global search ---
        # DE is robust to the discus function's narrow valley structure
        # Use a moderate population and generations to balance exploration/exploitation
        bounds = [(-5, 5)] * self.dim
        pop_size = 30
        max_iter = 50

        def de_callback(xk, convergence):
            # Update best if DE finds something better
            nonlocal best_f, best_x
            if xk < best_f:
                best_f = xk
                best_x = np.clip(xk, -5, 5).copy()
            return time.time() > deadline - 40

        try:
            de_result = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-12,
                atol=1e-14,
                seed=42,
                polish=False,  # We'll do our own polish
                init='sobol',
                callback=de_callback,
                updating='immediate'
            )
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass

        # --- Phase 3: Multi-restart L-BFGS-B for local refinement ---
        # Use the best point from DE plus several diverse starts
        starts = [best_x.copy()]

        # Add perturbations around the best point at various scales
        for scale in [0.3, 0.8, 1.5, 2.5]:
            for _ in range(2):
                perturbed = best_x + rng.normal(0, scale, self.dim)
                starts.append(np.clip(perturbed, -5, 5))

        # Add a few globally diverse points
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Shuffle non-best starts
        starts = [starts[0]] + starts[1:]
        rng.shuffle(starts[1:])

        for start_x in starts:
            if call_count >= self.budget or time.time() > deadline - 15:
                break

            start_x = np.clip(start_x, -5, 5)

            try:
                res = minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 150}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 4: Final high-precision polish ---
        if best_x is not None and call_count < self.budget and time.time() < deadline - 5:
            # Nelder-Mead for final polishing
            try:
                res_nm = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass

            # One more L-BFGS-B pass if budget allows
            if call_count < self.budget and time.time() < deadline - 2:
                try:
                    res = minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 200}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
