"""Multi-start Nelder-Mead with adaptive basin exploration for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Multi-start Nelder-Mead with adaptive basin exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        history = []  # List of (x, f) tuples

        # Fixed seed for reproducibility
        rng = np.random.default_rng(12345)

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            history.append((x, f_val))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        bounds = [(-5.0, 5.0)] * dim

        # --- Phase 1: Initial Exploration with Sobol ---
        n_sobol = 24
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points_unit = sampler.random(n_sobol)
            sobol_points = sobol_points_unit * 10.0 - 5.0

            for x in sobol_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                wrapped_func(x)
        except Exception:
            for _ in range(n_sobol):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x)

        # --- Phase 2: Multi-start Nelder-Mead with adaptive basins ---
        # Sort history and use top points as starting candidates
        if history:
            sorted_hist = sorted(history, key=lambda item: item[1])
            # Take top 8 unique-ish points
            top_candidates = []
            for x, f in sorted_hist:
                if len(top_candidates) >= 8:
                    break
                # Check if this point is sufficiently different from existing candidates
                is_unique = True
                for xc in top_candidates:
                    if np.linalg.norm(x - xc) < 0.5:
                        is_unique = False
                        break
                if is_unique:
                    top_candidates.append(x)
        else:
            top_candidates = [best_x]

        # Also add some random points far from current best for diversity
        for _ in range(4):
            x_new = rng.uniform(-5.0, 5.0, dim)
            top_candidates.append(x_new)

        # Nelder-Mead options - aggressive convergence
        nm_options = {
            'maxiter': 200,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }

        # Process each candidate
        for x_start in top_candidates:
            if eval_count >= budget or time.time() > deadline:
                break

            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='Nelder-Mead',
                    options=nm_options
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            # After each successful local search, if we have budget,
            # generate new starts near the current best with decreasing radius
            if eval_count < budget and time.time() < deadline:
                # Adaptive perturbation: start with larger radius, shrink as we converge
                improvement = 1.0
                if len(history) > 2:
                    # Measure recent improvement
                    recent = history[-10:]
                    f_vals = [f for _, f in recent]
                    if len(f_vals) >= 2:
                        improvement = max(1e-8, f_vals[0] - f_vals[-1])

                # Radius scales with improvement magnitude
                base_radius = min(2.0, max(0.1, 0.5 * np.sqrt(improvement)))

                for _ in range(3):
                    if eval_count >= budget or time.time() > deadline:
                        break
                    x_pert = best_x + rng.normal(0, base_radius, dim)
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    try:
                        res = minimize(
                            wrapped_func,
                            x_pert,
                            method='Nelder-Mead',
                            options=nm_options
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # --- Phase 3: Final Polish with L-BFGS-B ---
        if eval_count < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 4: Additional Nelder-Mead from best with small perturbations ---
        if eval_count < budget and time.time() < deadline:
            for radius in [0.3, 0.1, 0.05]:
                if eval_count >= budget or time.time() > deadline:
                    break
                x_pert = best_x + rng.normal(0, radius, dim)
                x_pert = np.clip(x_pert, -5.0, 5.0)
                try:
                    res = minimize(
                        wrapped_func,
                        x_pert,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
