"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0
    best_result = None
    best_score = 0.0
    best_wh = (1.0, 1.0)

    # Precompute pair indices for overlaps
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    # Broader aspect ratio search
    aspect_ratios = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3])
    max_restarts_per_ar = 5

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
        if np.any(r < 1e-10):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def hex_init(w, h, restart):
        ar_wh = w / h
        if ar_wh >= 1.0:
            n_cols = int(np.round(np.sqrt(N * ar_wh)))
            n_rows = int(np.ceil(N / n_cols))
        else:
            n_rows = int(np.round(np.sqrt(N / ar_wh)))
            n_cols = int(np.ceil(N / n_rows))
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        cell_w = w / n_cols
        cell_h = h / n_rows
        hex_h = cell_w * np.sqrt(3) / 2 if cell_w < cell_h else cell_h
        est_r = 0.45 * min(cell_w, hex_h)

        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x_center = (col + 0.5) * cell_w
            y_center = (row + 0.5) * cell_h
            if row % 2 == 1:
                x_center += 0.5 * cell_w
                x_center = np.clip(x_center, est_r, w - est_r)
            xs[i] = x_center
            ys[i] = y_center

        if restart > 0:
            jitter_scale = 0.1 * min(cell_w, cell_h)
            np.random.seed(42 + restart * 7)
            xs += np.random.normal(0, jitter_scale, N)
            ys += np.random.normal(0, jitter_scale, N)
            xs = np.clip(xs, 0.01, w - 0.01)
            ys = np.clip(ys, 0.01, h - 0.01)

        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
        r0 = 0.40 * min_dist
        radii = np.full(N, r0)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = xs[i]
            x0[3 * i + 1] = ys[i]
            x0[3 * i + 2] = radii[i]
        return x0

    def make_epigraph_solver(w, h):
        """
        Epigraph formulation: variables are [x, y, r, t] (4N).
        Objective: minimize t - sum(r)  (equivalent to maximize sum(r) - t)
        Constraints (all >= 0, ineq type):
          - t - (x_i - r_i) >= 0   for all i  (left wall)
          - t - (w - x_i - r_i) >= 0  (right wall)
          - t - (y_i - r_i) >= 0   for all i  (bottom wall)
          - t - (h - y_i - r_i) >= 0  (top wall)
          - t - (dist_ij - r_i - r_j) >= 0  for all pairs
          - t - 1e-6 >= 0  (t non-negative, small lower bound)
          - r_i - 1e-10 >= 0  for all i
        """
        n_vars = 4 * N

        def unpack4(v):
            x = v[0::4]
            y = v[1::4]
            r = v[2::4]
            t = v[3::4]  # all same value, take first
            return x, y, r, t[0]

        def obj(v):
            x, y, r, t = unpack4(v)
            return t - np.sum(r)

        def obj_grad(v):
            g = np.zeros(n_vars)
            # d/dt = 1, d/dr_i = -1
            g[3] = 1.0  # t index is 3
            for i in range(N):
                g[3 * i + 2] = -1.0  # r_i index is 3*i+2
            return g

        def make_constraints():
            # Wall constraints: 4N constraints
            def wall_cons(v):
                x, y, r, t = unpack4(v)
                return np.concatenate([
                    t - (x - r),
                    t - (w - x - r),
                    t - (y - r),
                    t - (h - y - r)
                ])

            def wall_grad(v):
                x, y, r, t = unpack4(v)
                # Shape: (4N, 4N)
                grad = np.zeros((4 * N, n_vars))
                # For each constraint k (0..4N-1), d/dv
                # t - (x_i - r_i): d/dt=1, d/dx_i=-1, d/dr_i=+1
                for i in range(N):
                    # Left wall: constraint index i
                    grad[i, 3] = 1.0
                    grad[i, 3 * i] = -1.0
                    grad[i, 3 * i + 2] = 1.0
                    # Right wall: constraint index N+i
                    grad[N + i, 3] = 1.0
                    grad[N + i, 3 * i] = 1.0
                    grad[N + i, 3 * i + 2] = 1.0
                    # Bottom wall: constraint index 2N+i
                    grad[2 * N + i, 3] = 1.0
                    grad[2 * N + i, 3 * i + 1] = -1.0
                    grad[2 * N + i, 3 * i + 2] = 1.0
                    # Top wall: constraint index 3N+i
                    grad[3 * N + i, 3] = 1.0
                    grad[3 * N + i, 3 * i + 1] = 1.0
                    grad[3 * N + i, 3 * i + 2] = 1.0
                return grad

            # Overlap constraints: N*(N-1)/2 constraints
            def overlap_cons(v):
                x, y, r, t = unpack4(v)
                dx = x[idx_i] - x[idx_j]
                dy = y[idx_i] - y[idx_j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-14)
                return t - dist - r[idx_i] - r[idx_j]

            def overlap_grad(v):
                x, y, r, t = unpack4(v)
                dx = x[idx_i] - x[idx_j]
                dy = y[idx_i] - y[idx_j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-14)
                # d(dist)/dx_i = dx_i/dist, d(dist)/dx_j = -dx_i/dist
                # Constraint: t - dist - r_i - r_j >= 0
                # d/dt = 1, d/dx_i = -dx/dist, d/dx_j = +dx/dist, d/dr_i = -1, d/dr_j = -1
                grad = np.zeros((n_pairs, n_vars))
                # t column
                grad[:, 3] = 1.0
                # x_i
                grad[:, 3 * idx_i] = -dx / dist
                # x_j
                grad[:, 3 * idx_j] = dx / dist
                # y_i
                grad[:, 3 * idx_i + 1] = -dy / dist
                # y_j
                grad[:, 3 * idx_j + 1] = dy / dist
                # r_i
                grad[:, 3 * idx_i + 2] = -1.0
                # r_j
                grad[:, 3 * idx_j + 2] = -1.0
                return grad

            # t lower bound
            def t_lb_cons(v):
                x, y, r, t = unpack4(v)
                return t - 1e-8

            def t_lb_grad(v):
                g = np.zeros(n_vars)
                g[3] = 1.0
                return g

            # r positive
            def r_pos_cons(v):
                x, y, r, t = unpack4(v)
                return r - 1e-10

            def r_pos_grad(v):
                grad = np.zeros((N, n_vars))
                for i in range(N):
                    grad[i, 3 * i + 2] = 1.0
                return grad

            return [
                {'type': 'ineq', 'fun': wall_cons, 'jac': wall_grad},
                {'type': 'ineq', 'fun': overlap_cons, 'jac': overlap_grad},
                {'type': 'ineq', 'fun': t_lb_cons, 'jac': t_lb_grad},
                {'type': 'ineq', 'fun': r_pos_cons, 'jac': r_pos_grad},
            ]

        return obj, obj_grad, make_constraints, unpack4

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        obj, obj_grad, make_constraints, unpack4 = make_epigraph_solver(w, h)
        constraints = make_constraints()

        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 10:
                break

            x0_3 = hex_init(w, h, restart)
            # Expand to 4N by adding t = max violation estimate
            x, y, r = unpack(x0_3)
            # Estimate t from current violations
            t_est = 0.0
            t_est = max(t_est, np.max(r - x), np.max(x + r - w), np.max(r - y), np.max(y + r - h))
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-14)
            t_est = max(t_est, np.max(r[idx_i] + r[idx_j] - dist))
            t_est = max(t_est, 1e-4)

            v0 = np.zeros(4 * N)
            v0[0::4] = x
            v0[1::4] = y
            v0[2::4] = r
            v0[3::4] = t_est

            try:
                result = minimize(
                    obj, v0, jac=obj_grad, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-14}
                )
                x_r, y_r, r_r, t_r = unpack4(result.x)

                if validate(x_r, y_r, r_r, w, h):
                    score = np.sum(r_r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_r, y_r, r_r])
                        best_wh = (w, h)

                        # Polish pass
                        if time.time() < deadline - 12:
                            try:
                                result2 = minimize(
                                    obj, result.x, jac=obj_grad, method='SLSQP',
                                    constraints=constraints,
                                    options={'maxiter': 5000, 'ftol': 1e-16}
                                )
                                x2, y2, r2, t2 = unpack4(result2.x)
                                if validate(x2, y2, r2, w, h):
                                    score2 = np.sum(r2)
                                    if score2 > best_score:
                                        best_score = score2
                                        best_result = np.column_stack([x2, y2, r2])
                                        best_wh = (w, h)
                            except Exception:
                                pass
            except Exception:
                continue

    # Fallback
    if best_result is None:
        w = h = 1.0
        n_cols = 5
        n_rows = 5
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(cells) < N:
                    cx = (j + 0.5) * w / n_cols
                    cy = (i + 0.5) * h / n_rows
                    r = 0.45 * min(w / n_cols, h / n_rows)
                    cells.append((cx, cy, r))
        return np.array(cells)

    # Final polish with known best_wh
    if best_result is not None and time.time() < deadline - 5:
        w_best, h_best = best_wh
        obj, obj_grad, make_constraints, unpack4 = make_epigraph_solver(w_best, h_best)
        constraints = make_constraints()

        x_best, y_best, r_best = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        t_est = 0.0
        t_est = max(t_est, np.max(r_best - x_best), np.max(x_best + r_best - w_best),
                    np.max(r_best - y_best), np.max(y_best + r_best - h_best))
        dx = x_best[idx_i] - x_best[idx_j]
        dy = y_best[idx_i] - y_best[idx_j]
        dist = np.sqrt(dx * dx + dy * dy + 1e-14)
        t_est = max(t_est, np.max(r_best[idx_i] + r_best[idx_j] - dist))
        t_est = max(t_est, 1e-6)

        v0 = np.zeros(4 * N)
        v0[0::4] = x_best
        v0[1::4] = y_best
        v0[2::4] = r_best
        v0[3::4] = t_est

        try:
            result_final = minimize(
                obj, v0, jac=obj_grad, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 10000, 'ftol': 1e-16}
            )
            xf, yf, rf, tf = unpack4(result_final.x)
            if validate(xf, yf, rf, w_best, h_best):
                score_final = np.sum(rf)
                if score_final > best_score:
                    best_score = score_final
                    best_result = np.column_stack([xf, yf, rf])
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
