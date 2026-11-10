"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Global exploration using Sobol points to find a good starting basin.
    2. Local refinement using Powell's method (derivative-free, adaptive per-dim steps),
       effective for anisotropic functions like Bent Cigar.
    3. A custom derivative-free trust-region / quadratic-surrogate solver that builds a
       local quadratic model from recent evaluations to capture the anisotropic curvature
       and drive rapid convergence in the narrow valley.
    4. A final high-precision polish pass with tighter tolerances.
    5. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # safety margin before the 330s kill
        d = self.dim
        budget = self.budget

        state = {"count": 0, "ok": True, "best_x": None, "best_f": np.inf}

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300

            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e300

            try:
                v = float(func(x))
            except Exception:
                return 1e300

            state["count"] += 1
            if not np.isfinite(v):
                return 1e300

            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x.copy()

            return v

        # --- Phase 1: Global Exploration (Sobol) ---
        n_sobol = min(20, budget // 10)

        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = -5.0 + 10.0 * sobol_points

            best_idx = 0
            best_val = np.inf

            for i in range(n_sobol):
                if not state["ok"]:
                    break
                x = sobol_points[i]
                v = wf(x)
                if v < best_val:
                    best_val = v
                    best_idx = i

            x0 = sobol_points[best_idx].copy()
        except Exception:
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: Local Refinement (Powell) ---
        remaining_budget = budget - state["count"]
        if remaining_budget > 0 and state["ok"]:
            try:
                res = minimize(
                    wf,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": 1000,
                        "maxfev": remaining_budget,
                        "xrtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 3: Derivative-free trust-region / quadratic-surrogate solver ---
        # Builds a local quadratic model from recent evaluations, fits it in a
        # trust region, and takes the steepest-descent direction of that model.
        # This directly models the anisotropic curvature of the Bent Cigar and
        # can converge much faster than a single (1+1)-ES step in the valley.
        if state["ok"] and state["best_x"] is not None:
            self._trust_region_solve(func, wf, state, d, budget, deadline)

        # --- Phase 4: Final high-precision polish pass ---
        remaining_budget = budget - state["count"]
        if remaining_budget > 5 and state["ok"] and state["best_x"] is not None:
            try:
                x0 = np.clip(state["best_x"], -5.0, 5.0)
                res = minimize(
                    wf,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": 500,
                        "maxfev": remaining_budget,
                        "xrtol": 1e-12,
                        "ftol": 1e-12,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        return state["best_x"], state["best_f"]

    # ------------------------------------------------------------------ #
    #  Derivative-free trust-region / quadratic-surrogate local solver    #
    # ------------------------------------------------------------------ #
    def _trust_region_solve(self, func, wf, state, d, budget, deadline):
        x = np.clip(np.asarray(state["best_x"], dtype=float), -5.0, 5.0).copy()
        fx = state["best_f"]

        # Trust-region radius: start relative to the domain scale.
        radius = 0.5

        # Recent-evaluation buffer used to build the quadratic model.
        # Store (point, value) pairs; keep at most M of them.
        M = 2 * d + 1
        pts = [x.copy()]
        vals = [fx]

        # Precompute a few candidate trust-region radii to try if the model
        # predicts a large step but the actual improvement is poor.
        radii = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        ridx = 0

        no_improve = 0
        max_outer = 200

        for outer in range(max_outer):
            if not state["ok"]:
                break
            if state["count"] >= budget:
                break
            if time.perf_counter() > deadline:
                state["ok"] = False
                break

            # Build the quadratic model from the recent buffer.
            try:
                direction, step = self._fit_quadratic_model(pts, vals, d, radius)
            except Exception:
                direction, step = None, 0.0

            if direction is None or step <= 0.0:
                # Fall back to a simple random perturbation to keep making progress.
                rng = np.random.default_rng(1000 + outer)
                direction = rng.standard_normal(d)
                step = radius

            # Propose a step within the trust region.
            step = min(step, radius)
            y = x + step * direction
            y = np.clip(y, -5.0, 5.0)

            fy = wf(y)

            if np.isfinite(fy) and fy < fx:
                # Accept: move to the new point.
                x = y
                fx = fy
                if fx < state["best_f"]:
                    state["best_f"] = fx
                    state["best_x"] = x.copy()

                # Update buffer.
                pts.append(x.copy())
                vals.append(fx)
                if len(pts) > M:
                    pts = pts[-M:]
                    vals = vals[-M:]

                # Expand the trust region modestly on success.
                ridx = max(0, ridx - 1)
                radius = radii[ridx]

                no_improve = 0
            else:
                # Reject: shrink the trust region.
                ridx = min(len(radii) - 1, ridx + 1)
                radius = radii[ridx]

                no_improve += 1
                if no_improve >= 12:
                    # Stuck at this scale; break out of the trust-region loop.
                    break

            # Convergence check: if the step is tiny relative to the domain, stop.
            if step < 1e-10:
                break

        # Make sure the buffer's best is recorded.
        if state["best_x"] is None:
            state["best_x"] = x.copy()
            state["best_f"] = fx

    def _fit_quadratic_model(self, pts, vals, d, radius):
        """
        Fit a local quadratic model q(z) = c + g.z + 0.5 z.H.z from the recent
        evaluations and return the steepest-descent direction and step length
        within the given trust radius.

        Returns (direction, step) or (None, 0.0) if the model is not usable.
        """
        n = len(pts)
        if n < d + 1:
            return None, 0.0

        # Center the data around the current best (last) point.
        c = np.asarray(pts[-1], dtype=float)
        X = np.array([np.asarray(p, dtype=float) - c for p in pts])
        y = np.array([float(v) for v in vals])

        # Build the design matrix for the quadratic model: [1, x_i, x_i.x_j]
        # Use only the lower-triangular set of quadratic terms to keep it square-ish.
        # Number of parameters: 1 + d + d*(d+1)/2
        nq = d * (d + 1) // 2
        npars = 1 + d + nq

        if n < npars:
            # Not enough points for a full quadratic; use a linear model instead.
            A = np.hstack([np.ones((n, 1)), X])
            b = y
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                return None, 0.0
            g = sol[1:]
            if not np.all(np.isfinite(g)):
                return None, 0.0
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-15:
                return None, 0.0
            direction = -g / gnorm
            step = radius
            return direction, step

        # Build the full quadratic design matrix.
        A = np.zeros((n, npars))
        A[:, 0] = 1.0
        A[:, 1:1 + d] = X
        # Fill in the quadratic terms.
        idx = 1 + d
        for i in range(d):
            for j in range(i, d):
                A[:, idx] = X[:, i] * X[:, j]
                idx += 1

        # Regularize the least-squares fit to keep the Hessian well-behaved.
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            return None, 0.0

        c0 = sol[0]
        g = sol[1:1 + d].copy()
        # Reconstruct the symmetric Hessian from the quadratic coefficients.
        H = np.zeros((d, d))
        idx = 1 + d
        for i in range(d):
            for j in range(i, d):
                H[i, j] = sol[idx]
                H[j, i] = sol[idx]
                idx += 1

        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(H)):
            return None, 0.0

        # Symmetrize and add a small ridge for numerical stability.
        H = 0.5 * (H + H.T)
        H += 1e-8 * np.eye(d)

        # Steepest descent direction of the quadratic model:
        #   min over step s of q(s * direction) along direction = -H^{-1} g
        # Solve H * d_step = g  ->  d_step = H^{-1} g, direction = -d_step.
        try:
            d_step = np.linalg.solve(H, g)
        except Exception:
            # Fallback: use the gradient direction.
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-15:
                return None, 0.0
            direction = -g / gnorm
            step = radius
            return direction, step

        dnorm = np.linalg.norm(d_step)
        if dnorm < 1e-15:
            return None, 0.0

        direction = -d_step / dnorm
        # Choose a step that minimizes the quadratic along this direction.
        # q(s) = c0 - s*(g.d) + 0.5 s^2 * (d^T H d), where d = direction.
        a = 0.5 * (direction @ (H @ direction))
        bcoef = -(g @ direction)
        if abs(a) > 1e-15:
            s_opt = -bcoef / (2.0 * a)
        else:
            s_opt = radius
        # Clip the step to the trust region.
        step = min(abs(s_opt), radius)
        if step <= 0.0:
            step = radius
        return direction, step
# EVOLVE-BLOCK-END
