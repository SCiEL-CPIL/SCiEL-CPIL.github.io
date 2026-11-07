"""Dual-phase optimizer for BBOB f19 (Griewank-Rosenbrock).

Phase 1: Dense Sobol quasi-random sequence to systematically map the
[-5,5]^5 space and locate the narrow Rosenbrock valley amidst Griewank
oscillations.

Phase 2: Multi-restart L-BFGS-B local refinement from the best Sobol
points. L-BFGS-B exploits gradient information for fast convergence in
the smooth Griewank-Rosenbrock landscape, especially in the mid-game
where single-trajectory methods struggle with the narrow valley.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Sobol global exploration + multi-restart L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_points(self, n_points, n, seed=0):
        """Generate quasi-random Sobol points in [0,1]^n."""
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=n, scramble=True, seed=seed)
            raw = sampler.random(n_points)  # shape (n_points, n)
        except Exception:
            # Fallback: Halton-like quasi-random
            raw = np.zeros((n_points, n))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for j in range(n):
                p = primes[j % len(primes)]
                for i in range(n_points):
                    f = 1.0
                    num = i + 1
                    while num > 0:
                        num, rem = divmod(num, p)
                        f /= p
                        raw[i, j] += rem * f
                    raw[i, j] = 1.0 - raw[i, j]
        return raw

    def _cmaes_fallback(self, x0, f0, func, budget, deadline, sigma0=1.0, pop=None):
        """Lightweight CMA-ES as fallback if L-BFGS-B fails."""
        n = self.dim
        if pop is None:
            pop = max(4, int(4 + 3 * np.log(n)))
        mu = max(2, pop // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs
        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        c = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_x = c.copy()
        best_f = f0
        calls = 0

        it = 0
        while calls < budget and time.time() < deadline:
            if time.time() > deadline:
                break
            Y = np.zeros((pop, n))
            F = np.empty(pop)
            for i in range(pop):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                x = np.clip(x, -5.0, 5.0)
                F[i] = float(func(x))
                Y[i] = y
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
            if calls >= budget:
                break
            idx = np.argsort(F)
            Ys = Y[idx]
            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean
            c = c + cs * d
            c = np.clip(c, -5.0, 5.0)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
                hsig = 1.0
            else:
                hsig = 0.0
            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-4, 10.0))
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                Csqrt = (evecs * np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                Csqrt = np.eye(n)
            it += 1
            if it > 5 and np.allclose(C, np.eye(n)) and sigma < 1e-3:
                break
        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        n = self.dim
        self._f = func

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ============================================================
        # PHASE 1: Dense Sobol quasi-random global exploration
        # ============================================================
        # Use ~40% of budget for systematic mapping
        sobol_budget = int(self.budget * 0.40)  # 200 points
        sobol_budget = min(sobol_budget, self.budget - 30)

        # Generate Sobol points and map to [-5, 5]^n
        sobol_raw = self._sobol_points(sobol_budget, n, seed=42)
        sobol_points = sobol_raw * 10.0 - 5.0  # map to [-5, 5]

        # Evaluate all Sobol points
        for i in range(sobol_budget):
            if time.time() > deadline - 60:  # leave time for local phase
                break
            if calls >= self.budget:
                break
            evaluate(sobol_points[i])

        # ============================================================
        # PHASE 2: Multi-restart L-BFGS-B local refinement
        # ============================================================
        # Collect diverse starting points from Sobol results
        # Sort Sobol points by their function values
        sobol_values = []
        for i in range(sobol_budget):
            if i >= len(sobol_raw):
                break
            # We already evaluated these, but we need to track values
            # Actually we need to re-track - let's use the stored best points
            pass

        # Better approach: keep track of all evaluated points and values
        # Let's restart with proper tracking
        # Actually, let's just use the best Sobol point and generate
        # diverse restarts around promising regions

        # Collect top-K Sobol points (we need to re-evaluate or track)
        # Since we tracked best_f, let's generate restarts from best_x
        # and also from structured diverse points

        # Strategy: multi-restart L-BFGS-B from:
        # 1. The best Sobol point
        # 2. Points near the best (perturbed)
        # 3. Structurally diverse points in the box

        local_budget = self.budget - calls
        if local_budget <= 10 or time.time() > deadline - 30:
            return best_x, best_f

        # Generate diverse restart points
        restart_points = []

        # 1. Best point found
        if best_x is not None:
            restart_points.append(best_x.copy())

        # 2. Perturbations of best point (multiple scales)
        if best_x is not None:
            rng = np.random.default_rng(123)
            for scale in [0.5, 1.0, 2.0, 3.0]:
                for _ in range(2):
                    pert = best_x + scale * rng.standard_normal(n)
                    pert = np.clip(pert, -5.0, 5.0)
                    restart_points.append(pert)

        # 3. Structurally diverse points (corners, edges, center)
        structured = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * (-2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0][:n]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0][:n]),
            np.array([0.0, 3.0, 0.0, -3.0, 0.0][:n]),
            np.array([3.0, 0.0, -3.0, 0.0, 3.0][:n]),
            np.array([-3.0, 0.0, 3.0, 0.0, -3.0][:n]),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0][:n]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0][:n]),
        ]
        for sp in structured:
            sp = np.clip(sp, -5.0, 5.0)
            restart_points.append(sp)

        # 4. Random diverse points
        for _ in range(8):
            rp = rng.uniform(-5.0, 5.0, n)
            restart_points.append(rp)

        # Deduplicate (remove points too close to each other)
        deduped = []
        for rp in restart_points:
            rp = np.clip(rp, -5.0, 5.0)
            is_dup = False
            for dp in deduped:
                if np.linalg.norm(rp - dp) < 0.5:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(rp)
        restart_points = deduped

        # Sort restart points by proximity to best (closest first for early progress)
        if best_x is not None:
            restart_points.sort(key=lambda p: np.linalg.norm(p - best_x))

        # L-BFGS-B with gradient via finite differences
        # Each L-BFGS-B call uses ~(n+1) evaluations per gradient step
        # Limit total local budget

        # Allocate local budget across restarts
        n_restarts = min(len(restart_points), 20)
        # Give more budget to closer/better restarts
        # First few restarts get more budget

        for ri in range(n_restarts):
            if calls >= self.budget or time.time() > deadline - 10:
                break

            x0 = restart_points[ri].copy()
            x0 = np.clip(x0, -5.0, 5.0)

            # Check if we've already evaluated this point (approximate)
            # Just proceed - L-BFGS-B will evaluate new points

            # Determine budget for this restart
            # Earlier restarts get more budget
            frac = max(0.3, 1.0 - ri * 0.08)
            restart_budget = max(10, int(local_budget * frac * 0.3))
            restart_budget = min(restart_budget, self.budget - calls)

            if restart_budget < 5:
                break

            # L-BFGS-B
            try:
                def objective(x):
                    x = np.clip(x, -5.0, 5.0)
                    return float(func(x))

                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max(5, restart_budget // (n + 1)),
                        'maxfun': restart_budget,
                        'maxls': 20,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'gtol_abs': 1e-10,
                        'disp': False,
                    }
                )

                if res.success or res.fun < best_f:
                    rx = np.clip(res.x, -5.0, 5.0)
                    rf = float(res.fun)
                    if rf < best_f:
                        best_f = rf
                        best_x = rx.copy()

            except Exception:
                pass

            # Check time
            if time.time() > deadline - 15:
                break

        # ============================================================
        # PHASE 3: Final CMA-ES polish if budget remains
        # ============================================================
        remaining = self.budget - calls
        if remaining > 20 and time.time() < deadline - 10 and best_x is not None:
            try:
                x3, f3, c3 = self._cmaes_fallback(
                    best_x, best_f, func, remaining, deadline,
                    sigma0=0.5, pop=6
                )
                if f3 < best_f:
                    best_f = f3
                    best_x = x3.copy()
            except Exception:
                pass

        # Final safety: ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))

        return best_x, best_f

# EVOLVE-BLOCK-END
