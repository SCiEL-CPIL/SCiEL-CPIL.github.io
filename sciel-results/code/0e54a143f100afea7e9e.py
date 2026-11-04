"""
Circle packing in unit square.
Combines:
 - direct (x,y,r) smooth-penalty optimization (radii optimized jointly),
 - multi-start diversity (grid/hex/random/voronoi) + simulated annealing,
 - best-fit local search, radius-growth / exact radius polish,
 - soft-min log-sum-exp (x,y)-only gradient polish,
 - final high-precision exact local solve (fixed-point radius update).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def construct():
    global _START
    _START = time.time()

    # ------------------------------------------------------------------
    # radius-from-centers (Voronoi inradius) helper
    # ------------------------------------------------------------------
    def compute_radii(x, y):
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        return np.maximum(r, 1e-6)

    # ------------------------------------------------------------------
    # validity / score
    # ------------------------------------------------------------------
    def check_validity(x, y, r, tol=1e-9):
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        need = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(need, 0.0)
        if np.any(dist < need - tol):
            return False
        return True

    def _score(x, y, r):
        if check_validity(x, y, r):
            return np.sum(r)
        for scale in [0.999, 0.999, 0.99, 0.98, 0.95, 0.90]:
            rt = r * scale
            if check_validity(x, y, rt):
                return np.sum(rt)
        return -np.inf

    # ------------------------------------------------------------------
    # direct (x,y,r) smooth penalty objective + analytic gradient
    # ------------------------------------------------------------------
    def penalty_obj_grad(v, mu):
        n = len(v) // 3
        pts = v.reshape(n, 3)
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

        obj = -np.sum(rs)

        # boundary violations
        b_left = np.maximum(0.0, rs - xs)
        b_bot = np.maximum(0.0, rs - ys)
        b_right = np.maximum(0.0, rs - (1.0 - xs))
        b_top = np.maximum(0.0, rs - (1.0 - ys))
        bv = b_left + b_bot + b_right + b_top
        bpen = np.sum(bv * bv)

        # overlap violations
        ii, jj = np.triu_indices(n, k=1)
        dx = xs[ii] - xs[jj]
        dy = ys[ii] - ys[jj]
        dist = np.sqrt(dx * dx + dy * dy)
        ov = np.maximum(0.0, rs[ii] + rs[jj] - dist)
        open_ = np.sum(ov * ov)

        energy = obj + mu * (bpen + open_)

        grad = np.zeros_like(v)
        grad[2::3] = -1.0

        db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
        db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
        db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
                  np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))
        grad[0::3] += mu * 2 * bv * db_dxi
        grad[1::3] += mu * 2 * bv * db_dyi
        grad[2::3] += mu * 2 * bv * db_dri

        mask = ov > 0
        if np.any(mask):
            im = ii[mask]
            jm = jj[mask]
            ov_m = ov[mask]
            dist_m = dist[mask]
            safe = np.where(dist_m < 1e-12, 1.0, dist_m)
            ux = dx[mask] / safe
            uy = dy[mask] / safe
            np.add.at(grad, 3 * im, -mu * 2 * ov_m * ux)
            np.add.at(grad, 3 * im + 1, -mu * 2 * ov_m * uy)
            np.add.at(grad, 3 * im + 2, mu * 2 * ov_m)
            np.add.at(grad, 3 * jm, mu * 2 * ov_m * ux)
            np.add.at(grad, 3 * jm + 1, mu * 2 * ov_m * uy)
            np.add.at(grad, 3 * jm + 2, mu * 2 * ov_m)

        return energy, grad

    def project_feasible(v, max_iter=800):
        n = len(v) // 3
        pts = v.reshape(n, 3).copy()
        ii, jj = np.triu_indices(n, k=1)
        for _ in range(max_iter):
            if _remaining() < 3.0:
                break
            xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
            d_left, d_bot = xs, ys
            d_right, d_top = 1.0 - xs, 1.0 - ys
            min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
            too_big = rs > min_d
            if np.any(too_big):
                rs[too_big] = np.maximum(1e-4, min_d[too_big] * (1 - 1e-9))
                pts[:, 2] = rs
            dx = xs[ii] - xs[jj]
            dy = ys[ii] - ys[jj]
            dist = np.sqrt(dx * dx + dy * dy)
            ov = np.maximum(0.0, rs[ii] + rs[jj] - dist)
            if not np.any(ov > 1e-10):
                break
            per = np.zeros(n)
            np.add.at(per, ii, ov)
            np.add.at(per, jj, ov)
            rs = np.maximum(1e-4, rs - per / 2.0 - 1e-10)
            pts[:, 2] = rs
        return pts.ravel()

    def direct_optimize(x0, y0, r0, max_mu=20000.0, max_iters=3000):
        v0 = np.concatenate([x0, y0, r0])
        v0 = np.clip(v0, 1e-3, 1.0)
        v0[2::3] = np.clip(v0[2::3], 1e-4, 0.49)
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
            bounds.append((1e-4, 0.49))
        best_v = v0.copy()
        best_val = _score(v0[:N], v0[N:2 * N], v0[2 * N:])
        mus = [10.0, 50.0, 200.0, 1000.0, 3000.0, max_mu]
        for mu in mus:
            if _remaining() < 8.0:
                break
            try:
                res = minimize(penalty_obj_grad, best_v, method='L-BFGS-B',
                               jac=True, args=(mu,), bounds=bounds,
                               options={'maxiter': max_iters, 'ftol': 1e-13,
                                        'gtol': 1e-10, 'maxls': 80})
                cand = res.x
            except Exception:
                continue
            cand = project_feasible(cand)
            cx, cy, cr = cand[:N], cand[N:2 * N], cand[2 * N:]
            val = _score(cx, cy, cr)
            if val > best_val:
                best_val = val
                best_v = cand
        return best_v[:N], best_v[N:2 * N], best_v[2 * N:], best_val

    # ------------------------------------------------------------------
    # (x,y)-only soft-min log-sum-exp objective (radii = inradius)
    # ------------------------------------------------------------------
    def smooth_xy(params, beta):
        x = params[:N]
        y = params[N:2 * N]
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy + 1e-30)
        np.fill_diagonal(dist, 1e10)
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)
        nbc = -beta * c
        mx = np.max(nbc, axis=1, keepdims=True)
        e = np.exp(nbc - mx)
        s = np.sum(e, axis=1, keepdims=True)
        r = -(mx + np.log(s)) / beta
        r = r[:, 0]
        w = e / s
        ds = dist.copy()
        np.fill_diagonal(ds, 1.0)
        dc_dx_i = 0.5 * dx / ds
        dc_dy_i = 0.5 * dy / ds
        dc_dx_j = -0.5 * dx / ds
        dc_dy_j = -0.5 * dy / ds
        b4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        bb = 100.0
        nbb = -bb * b4
        mxb = np.max(nbb, axis=1, keepdims=True)
        eb = np.exp(nbb - mxb)
        sb = np.sum(eb, axis=1, keepdims=True)
        wb = eb / sb
        g_rbx = wb[:, 0] - wb[:, 1]
        g_rby = wb[:, 2] - wb[:, 3]
        np.fill_diagonal(dc_dx_i, g_rbx)
        np.fill_diagonal(dc_dy_i, g_rby)
        np.fill_diagonal(dc_dx_j, g_rbx)
        np.fill_diagonal(dc_dy_j, g_rby)
        dr_dx = np.sum(w * dc_dx_i, axis=1) + np.sum(w * dc_dx_j, axis=0) - np.diag(w) * np.diag(dc_dx_i)
        dr_dy = np.sum(w * dc_dy_i, axis=1) + np.sum(w * dc_dy_j, axis=0) - np.diag(w) * np.diag(dc_dy_i)
        grad = np.concatenate([-dr_dx, -dr_dy])
        return -np.sum(r), grad

    def xy_polish(x0, y0, betas, max_iters=600):
        p = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        best = None
        best_val = -np.inf
        for beta in betas:
            if _remaining() < 6.0:
                break
            try:
                res = minimize(smooth_xy, p, method='L-BFGS-B', jac=True,
                               args=(beta,), bounds=bounds,
                               options={'maxiter': max_iters, 'ftol': 1e-13,
                                        'gtol': 1e-11, 'maxls': 80})
                p = res.x
            except Exception:
                continue
            r = compute_radii(p[:N], p[N:])
            val = _score(p[:N], p[N:], r)
            if val > best_val:
                best_val = val
                best = (p[:N].copy(), p[N:].copy(), r.copy(), val)
        if best is not None:
            return best
        return None

    # ------------------------------------------------------------------
    # radius refinement
    # ------------------------------------------------------------------
    def grow_radii(x, y, r, max_iters=600, step=0.0005):
        x = x.copy(); y = y.copy(); r = r.copy()
        for _ in range(max_iters):
            if _remaining() < 3.0:
                break
            improved = False
            for i in range(N):
                rnew = r.copy()
                rnew[i] += step
                if rnew[i] > 0.5:
                    continue
                if check_validity(x, y, rnew):
                    r[i] = rnew[i]
                    improved = True
            if not improved:
                break
        return r

    def exact_radius_polish(x, y, r, max_iters=100):
        """Fixed-point: r_i = min(boundary_i, 0.5*min_j dist_ij)."""
        x = x.copy(); y = y.copy(); r = r.copy()
        for _ in range(max_iters):
            if _remaining() < 3.0:
                break
            r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(dist, np.inf)
            r_new = np.minimum(r_bound, 0.5 * np.min(dist, axis=1))
            r_new = np.maximum(r_new, 1e-6)
            if np.max(np.abs(r_new - r)) < 1e-14:
                r = r_new
                break
            r = r_new
        return r

    # ------------------------------------------------------------------
    # initializations
    # ------------------------------------------------------------------
    def make_init(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N); y = np.zeros(N); idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N); y = np.zeros(N); idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def voronoi_init(seed):
        rng = np.random.RandomState(seed)
        n_grid = 12
        gx, gy = np.meshgrid(np.linspace(0.1, 0.9, n_grid), np.linspace(0.1, 0.9, n_grid))
        gx = gx.flatten(); gy = gy.flatten()
        bx = [0.5, 0.5, 0.1, 0.9, 0.1, 0.9, 0.5, 0.5]
        by = [0.1, 0.9, 0.1, 0.9, 0.5, 0.5, 0.1, 0.9]
        ax = np.concatenate([gx, bx]); ay = np.concatenate([gy, by])
        sx = []; sy = []
        rx = ax.copy(); ry = ay.copy()
        for _ in range(N):
            if len(rx) == 0:
                break
            if len(sx) == 0:
                idx = np.argmin((rx - 0.5) ** 2 + (ry - 0.5) ** 2)
            else:
                md = np.min(np.sqrt((rx[:, None] - np.array(sx)[None, :]) ** 2 +
                                    (ry[:, None] - np.array(sy)[None, :]) ** 2), axis=1)
                idx = np.argmax(md)
            sx.append(rx[idx]); sy.append(ry[idx])
            rx = np.delete(rx, idx); ry = np.delete(ry, idx)
        x = np.array(sx); y = np.array(sy)
        x += rng.uniform(-0.02, 0.02, N); y += rng.uniform(-0.02, 0.02, N)
        x = np.clip(x, 0.05, 0.95); y = np.clip(y, 0.05, 0.95)
        return x, y

    # ------------------------------------------------------------------
    # simulated annealing on (x,y), radii = inradius
    # ------------------------------------------------------------------
    def sa_optimize(x0, y0, seed, max_iters=3000):
        rng = np.random.RandomState(seed)
        x = x0.copy(); y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r); best_x = x.copy(); best_y = y.copy()
        T0 = 0.05; T1 = 0.0003
        cool = (T1 / T0) ** (1.0 / max_iters)
        rep = 2.0; bnd = 5.0
        for it in range(max_iters):
            if _remaining() < 5.0:
                break
            T = T0 * (cool ** it)
            r_curr = compute_radii(x, y)
            curr = np.sum(r_curr)
            fx = np.zeros(N); fy = np.zeros(N)
            fx += bnd * np.clip(0.1 - x, 0, None); fx -= bnd * np.clip(x - 0.9, 0, None)
            fy += bnd * np.clip(0.1 - y, 0, None); fy -= bnd * np.clip(y - 0.9, 0, None)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, 1e10)
            fm = rep / (d * d + 0.01)
            np.fill_diagonal(fm, 0)
            fx += np.sum(-fm * dx / d, axis=1)
            fy += np.sum(-fm * dy / d, axis=1)
            scale = 0.02 * T / T0 + 0.004
            dxp = fx * scale + rng.normal(0, scale, N)
            dyp = fy * scale + rng.normal(0, scale, N)
            xn = np.clip(x + dxp, 0.05, 0.95)
            yn = np.clip(y + dyp, 0.05, 0.95)
            rn = compute_radii(xn, yn)
            nv = np.sum(rn)
            delta = nv - curr
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = xn; y = yn; r = rn
                if nv > best_val:
                    best_val = nv; best_x = x.copy(); best_y = y.copy()
        return best_x, best_y, compute_radii(best_x, best_y), best_val

    # ------------------------------------------------------------------
    # best-fit local search
    # ------------------------------------------------------------------
    def best_fit(x, y, max_rounds=80):
        steps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                         [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)],
                         [-1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), -1 / np.sqrt(2)]])
        x = x.copy(); y = y.copy()
        r = compute_radii(x, y)
        curr = np.sum(r)
        for _ in range(max_rounds):
            if _remaining() < 5.0:
                break
            best_imp = 1e-10; best_i = -1; best_d = None; best_s = None
            for s in steps:
                xn = x[:, None] + dirs[None, :, 0] * s
                yn = y[:, None] + dirs[None, :, 1] * s
                valid = (xn >= 0.01) & (xn <= 0.99) & (yn >= 0.01) & (yn <= 0.99)
                for i in range(N):
                    for d in range(8):
                        if not valid[i, d]:
                            continue
                        xt = x.copy(); yt = y.copy()
                        xt[i] = xn[i, d]; yt[i] = yn[i, d]
                        rn = compute_radii(xt, yt)
                        imp = np.sum(rn) - curr
                        if imp > best_imp:
                            best_imp = imp; best_i = i; best_d = d; best_s = s
                if best_imp > s * 0.1:
                    break
            if best_i >= 0:
                x[best_i] += dirs[best_d, 0] * best_s
                y[best_i] += dirs[best_d, 1] * best_s
                r = compute_radii(x, y)
                curr = np.sum(r)
            else:
                break
        r = compute_radii(x, y)
        return x, y, r, np.sum(r)

    # ------------------------------------------------------------------
    # per-start refinement pipeline
    # ------------------------------------------------------------------
    def refine(x, y, r):
        # exact radius polish from current centers
        r = exact_radius_polish(x, y, r)
        val = _score(x, y, r)

        # best-fit on centers
        xb, yb, rb, vb = best_fit(x, y)
        if vb > val:
            x, y, r, val = xb, yb, rb, vb

        # direct (x,y,r) optimization
        if _remaining() > 12.0:
            xd, yd, rd, vd = direct_optimize(x, y, r)
            if vd > val:
                x, y, r, val = xd, yd, rd, vd

        # radius growth
        if _remaining() > 6.0:
            rg = grow_radii(x, y, r, max_iters=400)
            vg = _score(x, y, rg)
            if vg > val:
                r, val = rg, vg

        # soft-min (x,y) polish
        if _remaining() > 12.0:
            pol = xy_polish(x, y, betas=[50.0, 200.0, 1000.0, 5000.0], max_iters=800)
            if pol is not None:
                xp, yp, rp, vp = pol
                if vp > val:
                    x, y, r, val = xp, yp, rp, vp

        # final exact radius polish + growth
        r = exact_radius_polish(x, y, r)
        val = _score(x, y, r)
        if _remaining() > 4.0:
            rg = grow_radii(x, y, r, max_iters=300)
            vg = _score(x, y, rg)
            if vg > val:
                r, val = rg, vg
        return x, y, r, val

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    side = int(np.ceil(np.sqrt(N)))
    bx = np.zeros(N); by = np.zeros(N); idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            bx[idx] = (i + 0.5) / side
            by[idx] = (j + 0.5) / side
            idx += 1
    br = compute_radii(bx, by)
    best_val = _score(bx, by, br)

    modes = ['grid', 'hex', 'random', 'voronoi']
    num_starts = 28
    for i in range(num_starts):
        if _remaining() < 25.0:
            break
        mode = modes[i % len(modes)]
        if mode == 'voronoi':
            x0, y0 = voronoi_init(i)
        else:
            x0, y0 = make_init(i, mode)

        # SA to get good centers
        xs, ys, rs, vs = sa_optimize(x0, y0, seed=i, max_iters=2500)
        r = exact_radius_polish(xs, ys, rs)
        val = _score(xs, ys, r)
        if val > best_val:
            best_val = val; bx, by, br = xs.copy(), ys.copy(), r.copy()

        # full refinement
        if _remaining() > 40.0:
            xr, yr, rr, vr = refine(xs, ys, r)
            if vr > best_val:
                best_val = vr; bx, by, br = xr.copy(), yr.copy(), rr.copy()

    # final global polish on the best solution
    if _remaining() > 15.0:
        xf, yf, rf, vf = refine(bx, by, br)
        if vf > best_val:
            best_val = vf; bx, by, br = xf, yf, rf
        # second polish pass if time allows
        if _remaining() > 10.0:
            xf2, yf2, rf2, vf2 = refine(bx, by, br)
            if vf2 > best_val:
                best_val = vf2; bx, by, br = xf2, yf2, rf2

    # final validity safety
    if not check_validity(bx, by, br):
        for scale in np.linspace(0.99, 0.90, 20):
            rt = br * scale
            if check_validity(bx, by, rt):
                br = rt
                break
        else:
            br = compute_radii(bx, by)

    if _remaining() > 4.0:
        rg = grow_radii(bx, by, br, max_iters=300)
        if check_validity(bx, by, rg) and np.sum(rg) > np.sum(br):
            br = rg

    return np.column_stack((bx, by, br))
