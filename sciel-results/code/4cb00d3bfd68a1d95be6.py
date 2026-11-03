"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural change: joint (w, x, y, r) optimization with adaptive aspect-ratio
search, multi-seed initialization, contact-graph-aware Delaunay refinement, and
high-precision final polish. The rectangle width is a free variable (h = 2-w).
"""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None
    best_w = 1.0

    def _strictly_valid(circles, w, h, tol=1e-8):
        n = circles.shape[0]
        if n != N:
            return False
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
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
        nonlocal best_value, best_result, best_w
        if time_left() < 0.5:
            return
        val = _evaluate(circles, w, h)
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_w = w

    # Softplus helpers
    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0, x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    # ------------------------------------------------------------------
    # Objective: minimize negative sum of radii with soft constraint penalties
    # Variable layout: [w, x_0, y_0, r_0, x_1, y_1, r_1, ..., x_N-1, y_N-1, r_N-1]
    # h is derived: h = 2.0 - w
    # ------------------------------------------------------------------
    def _joint_obj_grad(x, beta=100.0, eq_weight=0.05, w_min=0.3, w_max=1.7):
        w = x[0]
        h = 2.0 - w
        c = x[1:].reshape(N, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        # Encourage equal radii (optional, small weight)
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[1 + 2::3] += -2.0 * eq_weight * r_dev  # d/d r_i

        # Wall constraints (softplus penalties)
        # Left: xs - rs >= 0
        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[1 + 0::3] -= g1  # d/d x_i
        grad[1 + 2::3] += g1  # d/d r_i

        # Right: xs + rs <= w
        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[0] += -np.sum(g2)  # d/d w
        grad[1 + 0::3] += g2
        grad[1 + 2::3] += g2

        # Bottom: ys - rs >= 0
        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[1 + 1::3] -= g3
        grad[1 + 2::3] += g3

        # Top: ys + rs <= h = 2 - w
        v4 = ys + rs - h  # = ys + rs - 2 + w
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[0] += np.sum(g4)  # d/d w: +g4 since h decreases with w
        grad[1 + 1::3] += g4
        grad[1 + 2::3] += g4

        # Positive radii: rs > 0
        v5 = -rs
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
        grad[1 + 2::3] -= g5

        # Pairwise non-overlap: |xi - xj| >= ri + rj
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            # d/d x_i
            grad[1 + 3 * i] += -np.sum(g * dx / d)
            grad[1 + 3 * i + 1] += -np.sum(g * dy / d)
            # d/d r_i
            grad[1 + 3 * i + 2] += np.sum(g)
            # d/d x_j, y_j, r_j
            idx = np.arange(i + 1, N)
            grad[1 + 3 * idx] += g * dx / d
            grad[1 + 3 * idx + 1] += g * dy / d
            grad[1 + 3 * idx + 2] += g

        # Maximize radii: constant negative gradient on r
        grad[1 + 2::3] += -1.0

        return obj, grad

    def _fixed_rect_obj_grad(x, w, h, beta=100.0, eq_weight=0.05):
        """x = [x_0, y_0, r_0, x_1, y_1, r_1, ...] with fixed w, h."""
        c = x.reshape(N, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)

        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[2::3] += -2.0 * eq_weight * r_dev

        v1 = rs - xs
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[0::3] -= g1
        grad[2::3] += g1

        v2 = xs + rs - w
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[0::3] += g2
        grad[2::3] += g2

        v3 = rs - ys
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[1::3] -= g3
        grad[2::3] += g3

        v4 = ys + rs - h
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[1::3] += g4
        grad[2::3] += g4

        v5 = -rs
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
        grad[2::3] -= g5

        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            grad[3 * i] += -np.sum(g * dx / d)
            grad[3 * i + 1] += -np.sum(g * dy / d)
            grad[3 * i + 2] += np.sum(g)
            idx = np.arange(i + 1, N)
            grad[3 * idx] += g * dx / d
            grad[3 * idx + 1] += g * dy / d
            grad[3 * idx + 2] += g

        grad[2::3] += -1.0
        return obj, grad

    def _hex_init(w, h, seed=42):
        """Generate a hexagonal-style initial packing."""
        rng = np.random.RandomState(seed)
        # Try several row configurations
        configs = [
            [3, 4, 4, 4, 3, 3], [4, 4, 4, 4, 3, 2], [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2], [4, 3, 4, 4, 4, 2], [2, 3, 4, 4, 4, 4],
            [3, 3, 4, 4, 4, 3], [4, 4, 3, 4, 4, 2], [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3], [5, 4, 4, 4, 4], [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5], [4, 4, 5, 4, 4],
        ]
        best = None
        best_val = -np.inf
        for row_counts in configs:
            if sum(row_counts) != N:
                continue
            n_rows = len(row_counts)
            max_cols = max(row_counts)
            est_r = min(w / (max_cols + 0.5), h / (n_rows * np.sqrt(3) / 2 + 1.0)) * 0.95
            if est_r < 0.01:
                continue
            circles = np.zeros((N, 3))
            idx = 0
            for row_idx, n_cols in enumerate(row_counts):
                y = (row_idx + 0.5) * est_r * np.sqrt(3)
                x_offset = est_r if row_idx % 2 == 1 else 0.0
                row_width = n_cols * 2 * est_r
                cx = w / 2
                for col_idx in range(n_cols):
                    x = cx - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * est_r
                    circles[idx, 0] = x
                    circles[idx, 1] = y
                    circles[idx, 2] = est_r
                    idx += 1
            if _strictly_valid(circles, w, h, tol=1e-6):
                val = np.sum(circles[:, 2])
                if val > best_val:
                    best_val = val
                    best = circles
        if best is None:
            # Fallback: grid
            a = int(np.ceil(np.sqrt(N * w / h)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            circles = np.zeros((N, 3))
            for i in range(N):
                row = i // a
                col = i % a
                circles[i, 0] = (col + 0.5) * w / a
                circles[i, 1] = (row + 0.5) * h / b
                circles[i, 2] = 0.5 * min(w / a, h / b) * 0.95
            best = circles
        return best

    def _random_init(w, h, seed):
        rng = np.random.RandomState(seed)
        r0 = min(w, h) / 2.0 * 0.3
        circles = np.zeros((N, 3))
        circles[:, 0] = rng.uniform(r0, w - r0, N)
        circles[:, 1] = rng.uniform(r0, h - r0, N)
        circles[:, 2] = r0 * rng.uniform(0.7, 1.0, N)
        return circles

    def _run_fixed_rect_polish(circles, w, h, beta_seq=None, max_iter=500):
        """Polish circles with fixed rectangle using L-BFGS-B."""
        if beta_seq is None:
            beta_seq = [10, 100, 1000, 5000]
        x = circles.flatten()
        best_x = x.copy()
        best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))
        r_max = min(w, h) / 2.0
        bounds = [(0, w), (0, h), (1e-6, r_max)] * N
        for beta in beta_seq:
            if time_left() < 3.0:
                break
            try:
                res = minimize(_fixed_rect_obj_grad, x, method='L-BFGS-B',
                               bounds=bounds, jac=True, args=(w, h, beta, 0.05),
                               options={'maxiter': max_iter, 'ftol': 1e-12,
                                        'gtol': 1e-12, 'maxls': 50})
                if time_left() < 2.0:
                    break
                nc = res.x.reshape(N, 3)
                if _strictly_valid(nc, w, h):
                    val = float(np.sum(nc[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x = res.x
            except Exception:
                pass
        return best_x.reshape(N, 3)

    def _run_joint_opt(w0, circles, beta_seq=None, max_iter=800):
        """Joint optimization of w and circle positions/radii."""
        if beta_seq is None:
            beta_seq = [10, 50, 200, 1000, 3000]
        h0 = 2.0 - w0
        x = np.concatenate([[w0], circles.flatten()])
        best_x = x.copy()
        best_val = -np.inf
        if _strictly_valid(circles, w0, h0):
            best_val = float(np.sum(circles[:, 2]))
        bounds = [(0.3, 1.7)] + [(0, 2.0), (0, 2.0), (1e-6, 0.8)] * N
        for beta in beta_seq:
            if time_left() < 4.0:
                break
            try:
                res = minimize(_joint_obj_grad, x, method='L-BFGS-B',
                               bounds=bounds, jac=True, args=(beta, 0.05, 0.3, 1.7),
                               options={'maxiter': max_iter, 'ftol': 1e-12,
                                        'gtol': 1e-12, 'maxls': 50})
                if time_left() < 2.5:
                    break
                w_new = res.x[0]
                h_new = 2.0 - w_new
                nc = res.x[1:].reshape(N, 3)
                if _strictly_valid(nc, w_new, h_new):
                    val = float(np.sum(nc[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
                x = res.x
            except Exception:
                pass
        w_final = best_x[0]
        h_final = 2.0 - w_final
        return w_final, best_x[1:].reshape(N, 3)

    def _delaunay_refine(circles, w, h, n_rounds=3):
        """Use Delaunay triangulation to identify close pairs and refine."""
        if time_left() < 5.0:
            return circles
        for _ in range(n_rounds):
            if time_left() < 3.0:
                break
            try:
                pts = circles[:, :2]
                tri = Delaunay(pts)
                # Find close pairs from Delaunay edges
                close_pairs = []
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            a, b = simplex[i], simplex[j]
                            d = np.hypot(circles[a, 0] - circles[b, 0],
                                         circles[a, 1] - circles[b, 1])
                            if d < circles[a, 2] + circles[b, 2] + 0.01:
                                close_pairs.append((a, b))
                if close_pairs:
                    # Slight perturbation for close pairs
                    for a, b in close_pairs:
                        dx = circles[b, 0] - circles[a, 0]
                        dy = circles[b, 1] - circles[a, 1]
                        d = np.hypot(dx, dy) + 1e-12
                        push = 0.002
                        circles[a, 0] -= push * dx / d
                        circles[a, 1] -= push * dy / d
                        circles[b, 0] += push * dx / d
                        circles[b, 1] += push * dy / d
            except Exception:
                pass
        return circles

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    # Phase 1: Grid search over width with hex init + fixed-rect polish
    widths = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    for w in widths:
        if time_left() < 8.0:
            break
        h = 2.0 - w
        init = _hex_init(w, h, seed=42)
        polished = _run_fixed_rect_polish(init, w, h, beta_seq=[10, 100, 500], max_iter=300)
        check_and_update(polished, w, h)

    # Phase 2: Joint optimization from best found so far
    if best_result is not None and time_left() > 15.0:
        w_j, c_j = _run_joint_opt(best_w, best_result, beta_seq=[10, 50, 200, 1000], max_iter=500)
        check_and_update(c_j, w_j, 2.0 - w_j)

    # Phase 3: Multiple random seeds with joint optimization
    for seed in [7, 13, 29, 53]:
        if time_left() < 10.0:
            break
        w_rand = best_w + 0.05 * np.random.RandomState(seed).choice([-1, 1])
        w_rand = np.clip(w_rand, 0.5, 1.5)
        h_rand = 2.0 - w_rand
        init = _random_init(w_rand, h_rand, seed)
        polished = _run_fixed_rect_polish(init, w_rand, h_rand, beta_seq=[10, 100, 500], max_iter=200)
        check_and_update(polished, w_rand, h_rand)
        if time_left() > 8.0:
            w_j, c_j = _run_joint_opt(w_rand, polished, beta_seq=[50, 200, 1000], max_iter=400)
            check_and_update(c_j, w_j, 2.0 - w_j)

    # Phase 4: High-precision final polish
    if best_result is not None and time_left() > 10.0:
        w_f = best_w
        h_f = 2.0 - w_f
        # Delaunay refine
        refined = _delaunay_refine(best_result.copy(), w_f, h_f, n_rounds=2)
        final = _run_fixed_rect_polish(refined, w_f, h_f, beta_seq=[1000, 5000, 10000], max_iter=2000)
        check_and_update(final, w_f, h_f)
        # Joint final polish
        if time_left() > 8.0:
            w_j, c_j = _run_joint_opt(w_f, final, beta_seq=[1000, 5000], max_iter=1000)
            check_and_update(c_j, w_j, 2.0 - w_j)

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        r = 0.08
        best_result = np.zeros((N, 3))
        a, b = 5, 5
        count = 0
        for i in range(a):
            for j in range(b):
                if count >= N:
                    break
                best_result[count, 0] = (i + 0.5) * w / a
                best_result[count, 1] = (j + 0.5) * h / b
                best_result[count, 2] = r
                count += 1

    return best_result
# EVOLVE-BLOCK-END
