"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead + L-BFGS-B optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline - 3

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                raise TimeoutError()
            if not time_ok():
                raise TimeoutError()
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Generate diverse starting points
        rng = np.random.default_rng(12345)
        starts = []

        # Center and structured points
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))
        starts.append(np.full(dim, 1.0))
        starts.append(np.full(dim, -1.0))

        # Quasi-random points using Halton-like sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, 40):
            pt = np.zeros(dim)
            for d in range(dim):
                base = d + 2  # 2, 3, 4, 5, 6
                pt[d] = (halton(i, base) - 0.5) * 8.0  # scale to [-4, 4]
            starts.append(pt)

        # Random points
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Sort: prefer center, then by norm
        starts.sort(key=lambda p: np.linalg.norm(p))

        # Phase 1: Quick Nelder-Mead from each start to find promising basins
        nm_results = []
        for x0 in starts:
            if not time_ok() or call_count >= budget - 50:
                break

            # Small budget per NM restart
            evals_per_nm = max(30, min(80, (budget - call_count) // max(1, len(starts))))
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': evals_per_nm,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
                nm_results.append(np.clip(result.x, -5, 5))
            except TimeoutError:
                break
            except Exception:
                pass

        # Phase 2: L-BFGS-B refinement from top candidates
        # Use gradient-free finite differences since we don't have gradient
        candidates = list(nm_results)
        if best_x is not None:
            candidates.insert(0, best_x.copy())

        # Deduplicate similar candidates
        unique_candidates = []
        for c in candidates:
            is_dup = False
            for uc in unique_candidates:
                if np.linalg.norm(c - uc) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_candidates.append(c)

        # Sort candidates by their function value (re-evaluate if needed)
        # Actually, we can just use the NM results order (already sorted by quality implicitly)

        for x0 in unique_candidates[:10]:  # Top 10 candidates
            if not time_ok() or call_count >= budget - 30:
                break

            remaining = budget - call_count
            if remaining < 50:
                break

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 500,
                        'maxfev': remaining,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except TimeoutError:
                break
            except Exception:
                pass

        # Phase 3: Final fine-tuning with Nelder-Mead from best point
        if best_x is not None and time_ok() and call_count < budget - 10:
            remaining = budget - call_count
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except (TimeoutError, Exception):
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
