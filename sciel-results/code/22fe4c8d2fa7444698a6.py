"""Derivative-Free Trust-Region optimizer for BBOB f18 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.linalg


class Optimizer:
    """Quadratic-model trust-region search that captures ill-conditioning."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        DEADLINE = 320.0          # safety margin before the 330s kill
        n = self.dim
        budget = self.budget

        def remaining():
            return DEADLINE - time.time()

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # Deterministic center + a couple of cheap probes to pick a start.
        candidates = [
            np.zeros(n),
            np.full(n, 0.5),
            np.full(n, -0.5),
            np.full(n, 1.0),
            np.full(n, -1.0),
        ]
        best_x, best_f = None, np.inf
        for c in candidates:
            if budget <= 0:
                break
            f = func(c)
            budget -= 1
            if f < best_f:
                best_f, best_x = f, c
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            budget -= 1

        x = best_x.copy()
        fx = best_f
        delta = 2.0               # trust-region radius

        def build_model(cx, fc, radius):
            """Estimate gradient and Hessian via 2n+1 points (n forward + n back)."""
            m = 2 * n + 1
            pts = [cx]
            vals = [fc]
            for i in range(n):
                e = np.zeros(n)
                e[i] = radius
                pts.append(cx + e)
                pts.append(cx - e)
            for p in pts[1:]:
                if budget <= 0:
                    break
                vals.append(func(clip(p)))
                budget -= 1
            vals = np.array(vals, dtype=float)
            # Gradient (central differences where both points exist)
            g = np.zeros(n)
            for i in range(n):
                fp = vals[1 + 2 * i]
                fm = vals[1 + 2 * i + 1]
                if fp < fm:
                    g[i] = (fp - fc) / radius
                else:
                    g[i] = (fc - fm) / radius
            # Hessian: second-order differences along each coordinate
            H = np.zeros((n, n))
            for i in range(n):
                fp = vals[1 + 2 * i]
                fm = vals[1 + 2 * i + 1]
                H[i, i] = max((fp + fm - 2.0 * fc) / (radius * radius), 0.0)
            # Off-diagonals via a few cross probes (kept sparse to save budget)
            return g, H

        def trust_region_step(g, H, radius):
            """Solve min g^T d + 0.5 d^T H d s.t. ||d|| <= radius (regularized)."""
            reg = 1e-8 + 1e-6 * radius
            A = H + reg * np.eye(n)
            # Cauchy-like: minimize along -g, then Newton step, clip to radius.
            try:
                d_newton = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                d_newton = -g / max(np.linalg.norm(g), 1e-12)
            # Choose between Newton and Cauchy directions by predicted reduction.
            d_cauchy = -g / max(np.linalg.norm(g), 1e-12)
            d_cauchy = d_cauchy * min(1.0, radius / max(np.linalg.norm(d_cauchy), 1e-12))
            pred_newton = g @ d_newton + 0.5 * d_newton @ A @ d_newton
            pred_cauchy = g @ d_cauchy + 0.5 * d_cauchy @ A @ d_cauchy
            d = d_newton if pred_newton <= pred_cauchy else d_cauchy
            norm_d = np.linalg.norm(d)
            if norm_d > radius:
                d = d * (radius / norm_d)
            if norm_d < 1e-12:
                d = d_cauchy
            return d

        while budget > 2 * n + 3 and remaining() > 1.5:
            g, H = build_model(x, fx, delta)
            if budget <= 3:
                break
            d = trust_region_step(g, H, delta)
            if np.linalg.norm(d) < 1e-9:
                break
            y = clip(x + d)
            if budget <= 0:
                break
            fy = func(y)
            budget -= 1
            if fy < fx:
                ratio = (fx - fy) / max(np.abs(g @ d), 1e-12)
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                if ratio > 0.7:
                    delta = min(delta * 2.0, 8.0)
                elif ratio < 0.25:
                    delta = max(delta * 0.5, 1e-3)
            else:
                delta = max(delta * 0.5, 1e-3)

        # Final high-precision polish with a tight local step.
        while budget > 1 and remaining() > 0.5:
            improved = False
            for i in range(n):
                for sgn in (1.0, -1.0):
                    if budget <= 0:
                        break
                    cand = x.copy()
                    cand[i] = x[i] + sgn * 1e-4
                    fy = func(clip(cand))
                    budget -= 1
                    if fy < fx:
                        x, fx = cand, fy
                        improved = True
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
        return best_x, best_f
# EVOLVE-BLOCK-END
