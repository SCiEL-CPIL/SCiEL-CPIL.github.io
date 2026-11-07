"""Deterministic multi-start optimizer for BBOB f19 (Griewank-Rosenbrock).

Structural design:
  Phase 1 - Stratified Latin-Hypercube global coverage (guarantees sampling of
            the rotated search space) + center + random diversity.
  Phase 2 - Accelerated BFGS local search from the best basins (exploits the
            smooth structure for rapid quadratic convergence).
  Phase 3 - Coordinate-descent fallback refinement (robustness against BFGS
            line-search failures).
  Phase 4 - Final BFGS polish from the global best.

Replaces stochastic evolutionary dynamics with a systematic global-to-local
refinement pipeline.
"""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start with stratified LHS + accelerated BFGS."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # explicit safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_call(x):
            """One budgeted function evaluation; updates best. Returns float or inf."""
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls += 1
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f):
                return np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def lhd_sample(n, dim, rng):
            """Latin Hypercube Design stratified over [-5, 5]."""
            samples = np.zeros((n, dim))
            for d in range(dim):
                strata = np.linspace(0.0, 1.0, n + 1)
                for i in range(n):
                    samples[i, d] = rng.uniform(strata[i], strata[i + 1])
                perm = rng.permutation(n)
                samples[:, d] = samples[perm, d]
            return samples * 10.0 - 5.0

        def bfgs(x0, max_evals=60):
            """Accelerated BFGS local search from x0 (finite-difference gradient)."""
            nonlocal best_x, best_f, calls
            if calls >= budget or time.time() > deadline:
                return
            x0 = np.clip(np.asarray(x0, dtype=np.float64), -5.0, 5.0)

            def objective(x):
                if calls >= budget or time.time() > deadline:
                    return np.inf
                return safe_call(x)

            try:
                res = minimize(objective, x0, method='BFGS',
                               options={'maxiter': max_evals, 'gtol': 1e-6,
                                        'maxls': 30})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        def coord_descent(x0, step=0.3, max_iter=40):
            """Robust coordinate-descent refinement (fallback for BFGS failures)."""
            nonlocal best_x, best_f, calls
            if calls >= budget or time.time() > deadline:
                return
            x = np.clip(np.asarray(x0, dtype=np.float64), -5.0, 5.0)
            f_cur = safe_call(x)
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for direction in (1, -1):
                        xt = x.copy()
                        xt[i] += direction * step
                        xt = np.clip(xt, -5.0, 5.0)
                        ft = safe_call(xt)
                        if ft < f_cur:
                            x = xt
                            f_cur = ft
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
            if f_cur < best_f:
                best_f = f_cur
                best_x = x.copy()

        # ---------------- Phase 1: global stratified coverage ----------------
        initial_points = [np.zeros(dim)]

        n_lhs = 25
        initial_points.extend([p for p in lhd_sample(n_lhs, dim, rng)])

        n_random = 10
        initial_points.extend([p for p in rng.uniform(-4.0, 4.0, size=(n_random, dim))])

        # Center first (cheap, often good on shifted/rotated BBOB), then by radius.
        initial_points.sort(key=lambda p: np.linalg.norm(p))

        evals = []
        for p in initial_points:
            if calls >= budget or time.time() > deadline:
                break
            f = safe_call(p)
            if np.isfinite(f):
                evals.append((f, p))

        # ---------------- Phase 2: BFGS from best basins ----------------
        if best_x is not None and calls < budget and time.time() < deadline:
            # Rank by fitness (already tracked best); refine top distinct starts.
            evals.sort(key=lambda t: t[0])
            seen = []
            candidates = []
            for f, p in evals:
                if len(candidates) >= 6:
                    break
                # keep distinct basins (simple distance gate)
                if any(np.linalg.norm(p - q) < 0.5 for q in seen):
                    continue
                seen.append(p)
                candidates.append(p)

            for cand in candidates:
                if calls >= budget or time.time() > deadline:
                    break
                if budget - calls < 8:
                    break
                bfgs(cand, max_evals=50)

        # ---------------- Phase 3: coordinate-descent fallback ----------------
        if best_x is not None and calls < budget and time.time() < deadline:
            coord_descent(best_x, step=0.3, max_iter=30)

        # ---------------- Phase 4: final BFGS polish ----------------
        if best_x is not None and calls < budget and time.time() < deadline:
            if budget - calls > 15:
                bfgs(best_x, max_evals=min(budget - calls, 40))

        if best_x is None:
            best_x = np.zeros(dim)
            if calls < budget:
                safe_call(best_x)

        return best_x, best_f
