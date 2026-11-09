"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize


class Optimizer:
    """Grid-based global search with local refinement for shifted/rotated
    Rosenbrock on [-5, 5]^5.  Uses a coarse lattice to find the basin, then
    Nelder-Mead + L-BFGS-B for refinement."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()

        # Deadline guard: return best before 330s kill
        def deadline():
            return time.perf_counter() - t0 > 260.0

        # Track all evaluations to respect budget
        call_count = 0
        best_x = None
        best_f = float("inf")

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return float("inf")
            call_count += 1
            try:
                f = float(func(x))
            except Exception:
                return float("inf")
            if f < best_f:
                best_f = f
                best_x = np.array(x, copy=True)
            return f

        # --- Phase 1: Coarse grid search over [-4, 4]^5 ----------------------
        # Use a 3x3x3x3x3 grid = 243 points to identify the basin region
        grid_vals = np.linspace(-4.0, 4.0, 5)  # 5 points per dim -> 5^5=3125, too many
        # Use 3 points per dim for coarse pass: 3^5 = 243
        grid_coarse = np.linspace(-4.0, 4.0, 3)
        grid_points = []
        for i0 in grid_coarse:
            for i1 in grid_coarse:
                for i2 in grid_coarse:
                    for i3 in grid_coarse:
                        for i4 in grid_coarse:
                            grid_points.append(np.array([i0, i1, i2, i3, i4]))

        # Evaluate all grid points (243 calls)
        for gp in grid_points:
            if call_count >= budget or deadline():
                break
            eval_func(gp)

        if deadline() or call_count >= budget:
            return best_x, best_f

        # --- Phase 2: Refine around the best grid point ----------------------
        # Take top few grid points and do local refinement
        # Re-evaluate to get sorted order (we already have best_f from eval_func)
        # Find the best grid point
        best_grid_x = best_x.copy() if best_x is not None else np.zeros(dim)

        # Generate refinement candidates around best grid point
        refinement_starts = [best_grid_x]
        
        # Add neighbors of best grid point (within grid)
        step = 4.0  # grid spacing for 3-point grid over [-4,4]
        for d in range(dim):
            for delta in (-step, step):
                candidate = best_grid_x.copy()
                candidate[d] += delta
                if np.all(candidate >= -5.0) and np.all(candidate <= 5.0):
                    refinement_starts.append(candidate)

        # Also add some deterministic points
        refinement_starts.append(np.zeros(dim))
        refinement_starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        refinement_starts.append(np.array([-1.0, -1.0, -1.0, -1.0, -1.0]))
        refinement_starts.append(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        refinement_starts.append(np.array([-2.0, -2.0, -2.0, -2.0, -2.0]))

        # Deduplicate
        unique_starts = []
        for s in refinement_starts:
            s = np.clip(s, -5.0, 5.0)
            dup = False
            for u in unique_starts:
                if np.array_equal(u, s):
                    dup = True
                    break
            if not dup:
                unique_starts.append(s)

        # --- Phase 3: Local refinement with L-BFGS-B and Nelder-Mead ---------
        for x0 in unique_starts:
            if deadline() or call_count >= budget - 50:
                break

            # L-BFGS-B refinement
            if call_count < budget - 10:
                try:
                    res = _scipy_minimize(
                        eval_func,
                        x0,
                        method="L-BFGS-B",
                        bounds=[(-5.0, 5.0)] * dim,
                        options={"maxiter": 50, "maxfun": 80,
                                 "gtol": 1e-6, "ftol": 1e-10},
                    )
                    rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                    eval_func(rx)
                except Exception:
                    pass

            if deadline() or call_count >= budget - 10:
                break

            # Nelder-Mead polish
            if call_count < budget - 5:
                try:
                    res = _scipy_minimize(
                        eval_func,
                        best_x if best_x is not None else x0,
                        method="Nelder-Mead",
                        bounds=[(-5.0, 5.0)] * dim,
                        options={"maxiter": 40, "maxfev": 40,
                                 "xatol": 1e-7, "fatol": 1e-10},
                    )
                    rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                    eval_func(rx)
                except Exception:
                    pass

        # Fallback in case nothing was evaluated
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = float("inf")

        return best_x, best_f
# EVOLVE-BLOCK-END
