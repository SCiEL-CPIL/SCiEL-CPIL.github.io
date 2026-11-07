"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B with Nelder-Mead polish for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin within 300s limit

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

        # --- Phase 1: Initial exploration (spread out calls early) ---
        # Evaluate a diverse set of points to find a good region quickly.
        n_explore = min(40, self.budget // 4)
        explored = []

        # Always include origin and axis-aligned points
        explored.append(np.zeros(self.dim))
        for d in range(self.dim):
            for s in [-3.0, 3.0]:
                p = np.zeros(self.dim)
                p[d] = s
                explored.append(p)

        # Random points at various scales
        for scale in [0.5, 1.5, 2.5, 3.5]:
            while len(explored) < n_explore:
                explored.append(rng.uniform(-scale, scale, self.dim))

        # Shuffle (but keep origin first)
        explored = [explored[0]] + explored[1:]
        rng.shuffle(explored[1:])

        for x in explored:
            if call_count >= self.budget or time.time() > deadline - 30:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(self.dim), wrapped_func(np.zeros(self.dim))

        # --- Phase 2: Local optimization from multiple diverse starts ---
        # Use the best point plus several other promising points as starts.
        starts = [best_x.copy()]

        # Add a few random perturbations around the best point
        for scale in [0.5, 1.0, 2.0]:
            for _ in range(3):
                perturbed = best_x + rng.normal(0, scale, self.dim)
                starts.append(np.clip(perturbed, -5, 5))

        # Add some globally diverse points (re-evaluate if not already seen)
        for scale in [1.0, 2.0, 3.0, 4.0]:
            for _ in range(2):
                starts.append(rng.uniform(-scale, scale, self.dim))

        # Shuffle the non-best starts
        starts = [starts[0]] + starts[1:]
        rng.shuffle(starts[1:])

        for start_x in starts:
            if call_count >= self.budget or time.time() > deadline - 20:
                break

            start_x = np.clip(start_x, -5, 5)

            # L-BFGS-B with tight tolerances
            try:
                res = minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            # Nelder-Mead polish from the current best
            if best_x is not None and call_count < self.budget and time.time() < deadline - 15:
                try:
                    res_nm = minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 60, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass

        # --- Phase 3: Final high-precision polish ---
        if best_x is not None and call_count < self.budget and time.time() < deadline - 5:
            # Try multiple polish strategies from the best point
            polish_starts = [best_x.copy()]

            # Small perturbations to escape potential flat regions
            for scale in [0.1, 0.05, 0.01]:
                for _ in range(2):
                    perturbed = best_x + rng.normal(0, scale, self.dim)
                    polish_starts.append(np.clip(perturbed, -5, 5))

            for px in polish_starts:
                if call_count >= self.budget or time.time() > deadline - 2:
                    break

                try:
                    res_final = minimize(
                        wrapped_func,
                        px,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True}
                    )
                    if res_final.fun < best_f:
                        best_f = res_final.fun
                        best_x = res_final.x.copy()
                except Exception:
                    pass

        # Final L-BFGS-B polish if budget remains
        if best_x is not None and call_count < self.budget and time.time() < deadline - 1:
            try:
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 300}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
