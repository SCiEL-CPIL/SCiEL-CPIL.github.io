"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural core: rigid equal-circle assembly on a fixed contact topology.
  - The only free variables are the 21 centers (positions); all radii are
    pinned to a single shared value r.
  - For a candidate (w, r) we project the centers onto the feasible set
    (walls + pairwise non-overlap) with a fast iterative solver and test
    whether a common r is achievable. Bisection over r finds the maximum.
  - A separate unequal-circle gradient polish (softplus penalties) is kept
    as a hybrid fallback to exploit any asymmetric optimum.
"""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None

    def _strictly_valid(circles, w, h, tol=1e-8):
        n = circles.shape[0]
        xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
        if np.any(rs < 1e-9):
            return False
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return False
        return True

    def _evaluate(circles, w, h, tol=1e-7):
        if not _strictly_valid(circles, w, h, tol):
            return -np.inf
        return float(np.sum(circles[:, 2]))

    def check_and_update(circles, w, h):
        nonlocal best_value, best_result
        if time_left() < 0.5:
            return
        val = _evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()

    # ------------------------------------------------------------------
    # RIGID EQUAL-CIRCLE ASSEMBLY  (structural core)
    # ------------------------------------------------------------------
    def _feasible(w, h, r, x0, iters=220, tol=1e-9):
        """Project centers onto the feasible set for a common radius r.
        Returns (feasible, centers) using a projected-gradient /
        Gauss-Seidel resolution of wall + pairwise constraints."""
        xs = x0[:, 0].astype(float).copy()
        ys = x0[:, 1].astype(float).copy()
        for _ in range(iters):
            # Wall clamps (exact projection onto the rectangle band)
            xs = np.clip(xs, r, w - r)
            ys = np.clip(ys, r, h - r)
            # Pairwise Gauss-Seidel: push apart any overlapping pair
            moved = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.hypot(dx, dy)
                    need = 2.0 * r
                    if d < need - tol:
                        if d < 1e-12:
                            dx, dy, d = 1e-6, 0.0, 1e-6
                        push = 0.5 * (need - d)
                        ux, uy = dx / d, dy / d
                        xs[i] += push * ux; ys[i] += push * uy
                        xs[j] -= push * ux; ys[j] -= push * uy
                        moved += push
            xs = np.clip(xs, r, w - r)
            ys = np.clip(ys, r, h - r)
            if moved < 1e-10:
                break
        # Final feasibility test
        xs = np.clip(xs, r, w - r); ys = np.clip(ys, r, h - r)
        if np.any(xs - r < -tol) or np.any(xs + r > w + tol):
            return False, np.column_stack([xs, ys])
        if np.any(ys - r < -tol) or np.any(ys + r > h + tol):
            return False, np.column_stack([xs, ys])
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = 2.0 * r - tol
            if np.any(d2 < need * need):
                return False, np.column_stack([xs, ys])
        return True, np.column_stack([xs, ys])

    def _hex_seed(w, h, r):
        """Hexagonal-lattice seed of N equal circles (positions only)."""
        best = None
        best_val = -np.inf
        row_configs = [
            [3, 4, 4, 4, 3, 3], [4, 4, 4, 4, 3, 2], [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2], [4, 3, 4, 4, 4, 2], [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3], [4, 4, 3, 4, 4, 2], [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3], [5, 4, 4, 4, 4], [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5], [4, 4, 5, 4, 4], [2, 5, 4, 5, 5],
            [3, 4, 5, 4, 5], [5, 4, 5, 4, 3],
        ]
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            est_r = min(w / (max_cols + 0.5), h / (n_rows * np.sqrt(3) / 2 + 1.0)) * 0.98
            if est_r < 0.01:
                continue
            circles = []
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * est_r * np.sqrt(3)
                x_offset = est_r if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * est_r
                cx = w / 2
                for col_idx in range(n_cols):
                    x = cx - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * est_r
                    circles.append([x, y])
            circles = np.array(circles)
            if np.all(circles[:, 0] > -0.5) and np.all(circles[:, 1] > -0.5):
                # seed quality: how much it fits before projection
                val = est_r
                if val > best_val:
                    best_val = val
                    best = circles
        if best is None:
            a = int(np.ceil(np.sqrt(N * w / h))); b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            best = np.array([[(i % a + 0.5) * w / a, (i // a + 0.5) * h / b] for i in range(N)])
        return best

    def _rigid_solve(w, h, seed):
        """Bisect the maximum common radius r for rectangle (w,h)."""
        if time_left() < 4.0:
            return None
        r_max = min(w, h) / 2.0
        # Upper bound: also limited by area-free packing heuristic
        r_hi = min(r_max, 0.28)
        # Find a feasible low point
        r_lo = 0.02
        feasible_lo, _ = _feasible(w, h, r_lo, _hex_seed(w, h, r_lo), iters=120)
        if not feasible_lo:
            return None
        seed_pos = _hex_seed(w, h, r_hi)
        best_r = r_lo
        best_c = None
        # Coarse bisection
        for _ in range(14):
            r_mid = 0.5 * (r_lo + r_hi)
            feas, pos = _feasible(w, h, r_mid, seed_pos, iters=200)
            if feas:
                best_r = r_mid
                best_c = np.column_stack([pos, np.full(N, r_mid)])
                r_lo = r_mid
            else:
                r_hi = r_mid
            if time_left() < 4.0:
                break
        # Fine bisection near the boundary
        for _ in range(10):
            r_mid = 0.5 * (r_lo + r_hi)
            feas, pos = _feasible(w, h, r_mid, best_c[:, :2] if best_c is not None else seed_pos, iters=260)
            if feas:
                best_r = r_mid
                best_c = np.column_stack([pos, np.full(N, r_mid)])
                r_lo = r_mid
            else:
                r_hi = r_mid
            if time_left() < 4.0:
                break
        if best_c is not None and _strictly_valid(best_c, w, h):
            return best_c
        return None

    # ------------------------------------------------------------------
    # UNEQUAL-CIRCLE GRADIENT POLISH  (hybrid fallback)
    # ------------------------------------------------------------------
    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0, x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
        c = x.reshape(-1, 3)
        xs = c[:, 0]; ys = c[:, 1]; rs = c[:, 2]
        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        if eq_weight > 0:
            r_mean = np.mean(rs); r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev
        v1 = rs - xs; obj += np.sum(_softplus(v1, beta)); g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1; grad[2::3] += g1
        v2 = xs + rs - w; obj += np.sum(_softplus(v2, beta)); g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2; grad[2::3] += g2
        v3 = rs - ys; obj += np.sum(_softplus(v3, beta)); g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3; grad[2::3] += g3
        v4 = ys + rs - h; obj += np.sum(_softplus(v4, beta)); g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4; grad[2::3] += g4
        v5 = -rs; obj += np.sum(_softplus(v5, beta)); g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5
        for i in range(N):
            dx = xs[i] - xs[i + 1:]; dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta)); g = _softplus_grad(v, beta)
            grad[3 * i]     += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx]     += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g
        grad[2::3] += -1.0
        return obj, grad

    def _unequal_init(w, h, seed):
        rng = np.random.RandomState(seed)
        r0 = min(w, h) / 2.0 * 0.25
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r0, w - r0, N)
        circles[:, 1] = rng.uniform(r0, h - r0, N)
        circles[:, 2] = r0 * rng.uniform(0.7, 1.0, N)
        return circles

    def _polish(circles, w, h, beta_start=10.0, max_iter=30):
        x = circles.flatten()
        best_x = x.copy(); best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = np.sum(circles[:, 2])
        r_max = min(w, h) / 2.0
        bounds = [(0, w), (0, h), (1e-4, r_max)] * N
        for beta in [beta_start, beta_start * 10, beta_start * 100, beta_start * 1000]:
            if time_left() < 3.0:
                break
            try:
                res = minimize(_objective_and_grad, x, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, 0.05),
                               options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10})
                if time_left() < 2.0:
                    break
                nc = res.x.reshape(-1, 3)
                if _strictly_valid(nc, w, h):
                    val = np.sum(nc[:, 2])
                    if val > best_val:
                        best_val = val; best_x = res.x.copy()
                x = res.x
            except Exception:
                pass
        return best_x.reshape(-1, 3)

    def _contact_polish(circles, w, h):
        if time_left() < 10.0:
            return circles
        x0 = circles.flatten()
        r_max = min(w, h) / 2.0
        bounds = [(0, w), (0, h), (1e-4, r_max)] * N
        best_x = x0.copy(); best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))
        for beta, eq_w, iters in [(300.0, 0.005, 2000), (800.0, 0.002, 3000),
                                  (2000.0, 0.001, 5000), (5000.0, 0.0005, 5000)]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(_objective_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, eq_w),
                               options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 80, 'maxfun': 50000})
                if time_left() < 3.0:
                    break
                nc = res.x.reshape(-1, 3)
                if _strictly_valid(nc, w, h):
                    val = float(np.sum(nc[:, 2]))
                    if val > best_val:
                        best_val = val; best_x = res.x.copy()
                x0 = res.x
            except Exception:
                pass
        return best_x.reshape(-1, 3)

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    for w in widths:
        if time_left() < 12.0:
            break
        h = 2.0 - w

        # (A) Rigid equal-circle assembly  (structural core)
        rigid = _rigid_solve(w, h, seed=42)
        if rigid is not None:
            check_and_update(rigid, w, h)

        # (B) Unequal-circle gradient polish  (hybrid fallback)
        if time_left() > 15.0:
            init = _unequal_init(w, h, seed=42)
            polished = _polish(init, w, h, beta_start=10.0, max_iter=30)
            check_and_update(polished, w, h)
            if time_left() > 12.0:
                cg = _contact_polish(polished, w, h)
                check_and_update(cg, w, h)

    # Fallback
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r

    return best_result
# EVOLVE-BLOCK-END
