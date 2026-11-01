import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vec_obj(z, penalty_weight=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w < 0.2 or h < 0.2 or np.any(r < 0.0):
        return 1e6
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)
    lv = np.maximum(r - x, 0.0)
    rv = np.maximum(x + r - w, 0.0)
    bv = np.maximum(r - y, 0.0)
    tv = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(lv*lv) + np.sum(rv*rv) + np.sum(bv*bv) + np.sum(tv*tv))
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2
    return obj


def _lse_obj(z, beta=60.0, pw=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w < 0.2 or h < 0.2 or np.any(r < 0.0):
        return 1e6
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    ov = r_sum[iu] - d_pairs
    s = np.clip(beta * ov, -500, 500)
    ms = np.max(s)
    obj += (500.0 / beta) * (ms + np.log(np.sum(np.exp(s - ms))))
    obj += 100.0 * np.sum(np.maximum(ov, 0.0) ** 2)
    viol = np.concatenate([r - x, x + r - w, r - y, y + r - h])
    sb = np.clip(beta * viol, -500, 500)
    mb = np.max(sb)
    obj += (500.0 / beta) * (mb + np.log(np.sum(np.exp(sb - mb))))
    obj += 100.0 * np.sum(np.maximum(viol, 0.0) ** 2)
    if w < 0.5:
        obj += 200.0 * (0.5 - w) ** 2
    elif w > 1.5:
        obj += 200.0 * (w - 1.5) ** 2
    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def unpack(z):
        return z[0::3][:N], z[1::3][:N], z[2::3][:N], z[-1]

    def is_valid(x, y, r, w, tol=1e-8):
        h = 2.0 - w
        if w <= tol or h <= tol:
            return False
        for i in range(N):
            if r[i] <= tol:
                return False
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                    return False
        return True

    def is_valid_eq(x, y, r, w, tol=1e-8):
        h = 2.0 - w
        if w <= tol or h <= tol or r <= tol:
            return False
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        if np.any(d_sq[iu] < (2 * r) ** 2 - tol):
            return False
        return True

    def sum_radii(r):
        return float(np.sum(r))

    def eq_obj(z, r, w):
        x = z[:N]
        y = z[N:2*N]
        h = 2.0 - w
        pen = 0.0
        pen += np.sum(np.maximum(r - x, 0.0) ** 2)
        pen += np.sum(np.maximum(x - (w - r), 0.0) ** 2)
        pen += np.sum(np.maximum(r - y, 0.0) ** 2)
        pen += np.sum(np.maximum(y - (h - r), 0.0) ** 2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-15))
        ov = np.maximum(2 * r - d_pairs, 0.0)
        pen += np.sum(ov ** 2)
        return pen

    def find_positions(r, w, x_init, y_init, max_iter=150):
        h = 2.0 - w
        z0 = np.concatenate([x_init, y_init])
        bounds = [(r, w - r)] * N + [(r, h - r)] * N
        try:
            res = minimize(eq_obj, z0, args=(r, w), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            return res.x[:N], res.x[N:2*N], res.fun
        except Exception:
            return x_init.copy(), y_init.copy(), np.inf

    def grid_init(w0, n=N):
        h0 = 2.0 - w0
        best_val = -1
        best = None
        for rows in range(3, 9):
            for cols in range(3, 9):
                if rows * cols < n or rows * cols > n + 8:
                    continue
                cell_w = w0 / cols
                cell_h = h0 / rows
                r_cell = min(cell_w, cell_h) / 2.0 * 0.98
                if r_cell < 0.015:
                    continue
                x = np.zeros(n); y = np.zeros(n); r = np.full(n, r_cell)
                idx = 0
                for ri in range(rows):
                    for ci in range(cols):
                        if idx >= n:
                            break
                        x[idx] = (ci + 0.5) * cell_w
                        y[idx] = (ri + 0.5) * cell_h
                        idx += 1
                for i in range(idx, n):
                    x[i] = w0 / 2.0; y[i] = h0 / 2.0; r[i] = r_cell * 0.5
                val = sum_radii(r)
                if val > best_val:
                    best_val = val
                    best = (x.copy(), y.copy(), r.copy())
        if best is None:
            cols = 5; rows = 5
            cell_w = w0 / cols; cell_h = h0 / rows
            r0 = min(cell_w, cell_h) / 2.0 * 0.9
            x = np.zeros(n); y = np.zeros(n); r = np.full(n, r0)
            for i in range(n):
                ri, ci = divmod(i, cols)
                x[i] = (ci + 0.5) * cell_w
                y[i] = (ri + 0.5) * cell_h
            best = (x, y, r)
        return best

    def hex_init(w0, n=N):
        h0 = 2.0 - w0
        area = w0 * h0
        r_est = min(np.sqrt(area / (n * np.pi / 0.9069)), 0.15)
        x = np.zeros(n); y = np.zeros(n); r = np.full(n, r_est)
        dx = 2.0 * r_est * 1.002
        dy = np.sqrt(3) * r_est * 1.002
        row = 0; col = 0; idx = 0
        while idx < n:
            cx = dx * (col + 0.5 * (row % 2)) + r_est
            cy = dy * row + r_est
            if cx > w0 - r_est:
                row += 1; col = 0
                cx = dx * (col + 0.5 * (row % 2)) + r_est
                cy = dy * row + r_est
            if cy > h0 - r_est:
                break
            x[idx] = cx; y[idx] = cy
            idx += 1; col += 1
        for i in range(idx, n):
            x[i] = w0 / 2.0; y[i] = h0 / 2.0; r[i] = r_est * 0.6
        return x, y, r

    def force_relax(x, y, r, w, max_iter=800, tol=1e-10):
        h = 2.0 - w
        x = x.copy().astype(float); y = y.copy().astype(float)
        dt = 0.003; min_dt = 0.00005
        for it in range(max_iter):
            fx = np.zeros(N); fy = np.zeros(N)
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 200
                if x[i] > w - r[i]:
                    fx[i] -= (x[i] - (w - r[i])) * 200
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 200
                if y[i] > h - r[i]:
                    fy[i] -= (y[i] - (h - r[i])) * 200
            max_ov = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]; dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    rs = r[i] + r[j]
                    ov = rs - d
                    if ov > 0:
                        ux = dx / d; uy = dy / d
                        f = ov * 30.0
                        fx[i] += ux * f; fy[i] += uy * f
                        fx[j] -= ux * f; fy[j] -= uy * f
                        max_ov = max(max_ov, ov)
            for i in range(N):
                max_ov = max(max_ov, r[i] - x[i], x[i] - (w - r[i]), r[i] - y[i], y[i] - (h - r[i]))
            if max_ov < tol:
                break
            fm = np.max(np.sqrt(fx * fx + fy * fy))
            if fm > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif fm < 0.05:
                dt = min(0.008, dt * 1.15)
            x += dt * fx; y += dt * fy
            for i in range(N):
                x[i] = np.clip(x[i], r[i], w - r[i])
                y[i] = np.clip(y[i], r[i], h - r[i])
        return x, y

    def to_z(x, y, r, w):
        z = np.zeros(3 * N + 1)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        z[-1] = w
        return z

    def from_z(z):
        return z[0::3][:N], z[1::3][:N], z[2::3][:N], z[-1]

    def slsqp(x, y, r, w, time_limit=12.0, use_lse=False):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return None
        z0 = to_z(x, y, r, w)
        maxiter = int(min(400, max(80, remaining * 25)))
        h0 = 2.0 - w
        bounds = []
        for i in range(N):
            bounds.append((r[i] * 0.3, w - r[i] * 0.3))
        for i in range(N):
            bounds.append((r[i] * 0.3, max(r[i] * 0.3, h0 - r[i] * 0.3)))
        for i in range(N):
            bounds.append((0.01, 0.45))
        bounds.append((0.35, 1.65))
        obj = _lse_obj if use_lse else _vec_obj
        try:
            res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 80})
            xo, yo, ro, wo = from_z(res.x)
            if np.any(ro < 0.005):
                return None
            if is_valid(xo, yo, ro, wo):
                return xo, yo, ro, wo
            xf, yf = force_relax(xo, yo, ro, wo, max_iter=200)
            if is_valid(xf, yf, ro, wo):
                return xf, yf, ro, wo
        except Exception:
            pass
        return None

    def relax_positions(x, y, r, w, max_iter=150):
        h = 2.0 - w
        def obj(z):
            xz = z[:N]; yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]; dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen
        z0 = np.concatenate([x, y])
        bounds = [(r[i], w - r[i]) for i in range(N)] + [(r[i], h - r[i]) for i in range(N)]
        try:
            res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
            return res.x[:N], res.x[N:2*N]
        except Exception:
            return x.copy(), y.copy()

    def binary_search_eq(w, x_init, y_init, r_low, r_high, max_iter=12, tol=1e-6):
        h = 2.0 - w
        bx, by, br = x_init.copy(), y_init.copy(), r_low
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            rm = (r_low + r_high) / 2.0
            if rm - r_low < tol:
                break
            xo, yo, ov = find_positions(rm, w, x_init, y_init, max_iter=120)
            if ov < 1e-8 and is_valid_eq(xo, yo, rm, w):
                bx, by, br = xo.copy(), yo.copy(), rm
                r_low = rm
            else:
                r_high = rm
        return bx, by, br

    def grow_eq(x, y, r, w, max_rounds=10):
        rc = r[0]
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            rn = rc * 1.005
            xo, yo, ov = find_positions(rn, w, x, y, max_iter=100)
            if ov < 1e-8 and is_valid_eq(xo, yo, rn, w):
                x, y, rc = xo, yo, rn
            else:
                break
        return x, y, np.full(N, rc), w

    def try_grow(x, y, r, w, max_rounds=5):
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 8:
                    break
                for _ in range(40):
                    rn = r.copy()
                    rn[i] *= 1.003
                    xt, yt = relax_positions(x, y, rn, w, max_iter=40)
                    if is_valid(xt, yt, rn, w):
                        x, y, r = xt, yt, rn
                        improved = True
                    else:
                        break
            if not improved:
                break
        return x, y, r, w

    def polish(x, y, r, w):
        h = 2.0 - w
        def obj(z):
            xz = z[:N]; yz = z[N:2*N]
            pen = 0.0
            for i in range(N):
                if xz[i] < r[i]:
                    pen += (r[i] - xz[i]) ** 2
                if xz[i] > w - r[i]:
                    pen += (xz[i] - (w - r[i])) ** 2
                if yz[i] < r[i]:
                    pen += (r[i] - yz[i]) ** 2
                if yz[i] > h - r[i]:
                    pen += (yz[i] - (h - r[i])) ** 2
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xz[i] - xz[j]; dy = yz[i] - yz[j]
                    d = np.sqrt(dx * dx + dy * dy) + 1e-12
                    ov = r[i] + r[j] - d
                    if ov > 0:
                        pen += ov * ov
            return pen
        z0 = np.concatenate([x, y])
        try:
            res = minimize(obj, z0, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-14})
            xt, yt = res.x[:N], res.x[N:2*N]
            if is_valid(xt, yt, r, w):
                return xt, yt, r, w
        except Exception:
            pass
        return x, y, r, w

    best_value = 0.0
    best_result = None
    rng = np.random.default_rng(42)

    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.7, 0.55, 1.1, 0.9, 1.25, 0.8, 1.4, 0.7, 1.6, 0.65]

    for w0 in aspect_ratios:
        if time.time() > deadline - 45:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
        for init in (grid_init, hex_init):
            x0, y0, r0 = init(w0)
            xt, yt = force_relax(x0, y0, r0, w0, max_iter=500)
            if is_valid(xt, yt, r0, w0):
                val = sum_radii(r0)
                if val > best_value:
                    best_value = val
                    best_result = (xt.copy(), yt.copy(), r0.copy(), w0)
            if time.time() < deadline - 40:
                r_iv = r0[0] if np.allclose(r0, r0[0]) else min(r0)
                xe, ye, re = binary_search_eq(w0, xt, yt, r_iv * 0.9, r_iv * 1.15, max_iter=10)
                if re > 0.01:
                    ra = np.full(N, re)
                    ve = sum_radii(ra)
                    if ve > best_value:
                        best_value = ve
                        best_result = (xe.copy(), ye.copy(), ra.copy(), w0)

    def refine(x, y, r, w):
        nonlocal best_value, best_result
        if time.time() < deadline - 12:
            res = slsqp(x, y, r, w, time_limit=10.0, use_lse=False)
            if res is not None:
                xs, ys, rs, ws = res
                vs = sum_radii(rs)
                if vs > best_value:
                    best_value = vs
                    best_result = (xs.copy(), ys.copy(), rs.copy(), ws)
        if time.time() < deadline - 8:
            res = slsqp(x, y, r, w, time_limit=8.0, use_lse=True)
            if res is not None:
                xs, ys, rs, ws = res
                vs = sum_radii(rs)
                if vs > best_value:
                    best_value = vs
                    best_result = (xs.copy(), ys.copy(), rs.copy(), ws)
        if time.time() < deadline - 6:
            if np.std(r) < 0.005:
                xe, ye, rea, we = grow_eq(x, y, r, w, max_rounds=6)
                if is_valid_eq(xe, ye, rea[0], we):
                    ve = sum_radii(rea)
                    if ve > best_value:
                        best_value = ve
                        best_result = (xe.copy(), ye.copy(), rea.copy(), we)
            else:
                xg, yg, rg, wg = try_grow(x, y, r, w, max_rounds=3)
                if is_valid(xg, yg, rg, wg):
                    vg = sum_radii(rg)
                    if vg > best_value:
                        best_value = vg
                        best_result = (xg.copy(), yg.copy(), rg.copy(), wg)
        if time.time() < deadline - 4:
            xp, yp, rp, wp = polish(x, y, r, w)
            if is_valid(xp, yp, rp, wp):
                vp = sum_radii(rp)
                if vp > best_value:
                    best_value = vp
                    best_result = (xp.copy(), yp.copy(), rp.copy(), wp)

    if best_result is not None:
        refine(*best_result)

    # random restarts
    if time.time() < deadline - 25:
        for _ in range(12):
            if time.time() > deadline - 20:
                break
            w0 = rng.uniform(0.55, 1.45)
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            if rng.random() < 0.5:
                x0, y0, r0 = grid_init(w0)
            else:
                x0, y0, r0 = hex_init(w0)
            x0 = np.clip(x0 + rng.normal(0, 0.008, N), r0, w0 - r0)
            y0 = np.clip(y0 + rng.normal(0, 0.008, N), r0, h0 - r0)
            xt, yt = force_relax(x0, y0, r0, w0, max_iter=300)
            if not is_valid(xt, yt, r0, w0):
                continue
            if sum_radii(r0) > best_value * 0.9:
                refine(xt, yt, r0, w0)

    if best_result is None:
        w0 = 1.0; h0 = 1.0; cols = 5; rows = 5
        cell_w = w0 / cols; cell_h = h0 / rows
        r0 = min(cell_w, cell_h) / 2.0 * 0.45
        cells = []
        for i in range(N):
            row, col = divmod(i, cols)
            cells.append(((col + 0.5) * cell_w, (row + 0.5) * cell_h, r0))
        return np.array(cells)

    xf, yf, rf, wf = best_result
    for _ in range(10):
        rt = rf * (1 - 1e-10)
        if is_valid(xf, yf, rt, wf):
            rf = rt
        else:
            break
    return np.column_stack((xf, yf, rf))
