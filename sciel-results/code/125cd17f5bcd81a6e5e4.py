import numpy as np
import time

# ---- constants from the task ----
N = 8
L = 1.0 / N          # tube side length
DEADLINE = 620.0     # safety margin under the 630 s wall-clock kill


def _base_positions():
    """Current separable 1D solution (known good, score ~1.3352)."""
    base = -np.arange(N) / (2.0 * N)
    ix, iy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    x = base[ix.ravel()]
    y = base[iy.ravel()]
    return np.stack([x, y], axis=1)


def _volume_exact(pos):
    """
    Deterministic exact union volume (correlates with the MC estimate).
    pos: (64, 2) array of (x_k, y_k).
    """
    x = pos[:, 0]
    y = pos[:, 1]
    i = np.repeat(np.arange(N), N)
    j = np.tile(np.arange(N), N)
    sx = i * L
    sy = j * L

    X0 = x.min()
    X1 = (x + (i + 1) * L).max()
    Y0 = y.min()
    Y1 = (y + (j + 1) * L).max()
    bw = X1 - X0
    bh = Y1 - Y0
    if bw <= 1e-9 or bh <= 1e-9:
        return np.inf

    # critical a / b values (tube left & right edges at c=0)
    ca = np.concatenate([x, x + L])
    cb = np.concatenate([y, y + L])
    ca = np.sort(np.unique(ca))
    cb = np.sort(np.unique(cb))

    a_mid = (ca[:-1] + ca[1:]) / 2.0
    b_mid = (cb[:-1] + cb[1:]) / 2.0
    da = np.diff(ca)
    db = np.diff(cb)

    # per-cell arrays: (nca, 64), (ncb, 64)
    aL = a_mid[:, None] - x[None, :] - sx[None, :]
    aR = a_mid[:, None] - x[None, :] - sx[None, :] - L
    bL = b_mid[:, None] - y[None, :] - sy[None, :]
    bR = b_mid[:, None] - y[None, :] - sy[None, :] - L

    invsx = 1.0 / np.where(sx > 1e-12, sx, 1.0)[None, :]
    invsy = 1.0 / np.where(sy > 1e-12, sy, 1.0)[None, :]
    hasx = (sx > 1e-12)[None, :]
    hasy = (sy > 1e-12)[None, :]

    # x-interval endpoints (clamped to [0,1]), shape (nca, 64)
    xlo = np.where(hasx, np.maximum(0.0, aL * invsx), 0.0)
    xhi = np.where(hasx, np.minimum(1.0, aR * invsx), 1.0)
    # y-interval endpoints, shape (ncb, 64)
    ylo = np.where(hasy, np.maximum(0.0, bL * invsy), 0.0)
    yhi = np.where(hasy, np.minimum(1.0, bR * invsy), 1.0)

    # union length of x-intervals at each a-row  -> (nca,)
    xl = xlo.ravel()
    xh = xhi.ravel()
    order = np.argsort(xl, kind="stable")
    xl_s = xl[order]
    xh_s = xh[order]
    valid = xh_s > xl_s
    xl_s = xl_s[valid]
    xh_s = xh_s[valid]
    if xl_s.size:
        cur = xl_s[0]
        end = xh_s[0]
        xun = np.empty(xl_s.shape)
        for k in range(1, xl_s.size):
            s = xl_s[k]
            e = xh_s[k]
            if s > end:
                xun[k - 1] = end - cur
                cur = s
                end = e
            else:
                if e > end:
                    end = e
        xun[xl_s.size - 1] = end - cur
        xun_row = xun.reshape(xlo.shape)
    else:
        xun_row = np.zeros(xlo.shape)

    # union length of y-intervals at each b-row -> (ncb,)
    yl = ylo.ravel()
    yh = yhi.ravel()
    order = np.argsort(yl, kind="stable")
    yl_s = yl[order]
    yh_s = yh[order]
    valid = yh_s > yl_s
    yl_s = yl_s[valid]
    yh_s = yh_s[valid]
    if yl_s.size:
        cur = yl_s[0]
        end = yh_s[0]
        yun = np.empty(yl_s.shape)
        for k in range(1, yl_s.size):
            s = yl_s[k]
            e = yh_s[k]
            if s > end:
                yun[k - 1] = end - cur
                cur = s
                end = e
            else:
                if e > end:
                    end = e
        yun[yl_s.size - 1] = end - cur
        yun_col = yun.reshape(ylo.shape)
    else:
        yun_col = np.zeros(ylo.shape)

    # hit fraction = (1/2) * sum over cells of xun_row[a]*yun_col[b]*da[a]*db[b]
    # = (1/2) * (xun_row * da) @ (yun_col * db)
    frac = 0.5 * float(np.dot(xun_row * da, yun_col * db))
    return frac * bw * bh


def _optimize(pos0, start):
    """Steepest-descent on 128 positions with a global box-shrink lever."""
    pos = pos0.copy()
    best = pos.copy()
    best_v = _volume_exact(pos)
    if not np.isfinite(best_v):
        return best, best_v
    n = pos.shape[0]

    # ---- finite-difference gradient ----
    def grad(p, h=1e-4):
        g = np.empty_like(p)
        for c in range(2):
            pp = p.copy()
            pp[:, c] += h
            vm = _volume_exact(pp)
            pp = p.copy()
            pp[:, c] -= h
            vp = _volume_exact(pp)
            g[:, c] = (vm - vp) / (2.0 * h)
        return g

    # ---- global box-shrink move: pull everything toward box center ----
    def shrink(p, alpha):
        x = p[:, 0]
        y = p[:, 1]
        i = np.repeat(np.arange(N), N)
        j = np.tile(np.arange(N), N)
        X0 = x.min()
        X1 = (x + (i + 1) * L).max()
        Y0 = y.min()
        Y1 = (y + (j + 1) * L).max()
        cx = 0.5 * (X0 + X1)
        cy = 0.5 * (Y0 + Y1)
        q = p.copy()
        q[:, 0] = x + alpha * (cx - x)
        q[:, 1] = y + alpha * (cy - y)
        return q

    # ---- per-coordinate box-shrink gradient (finite diff) ----
    def shrink_grad(p, h=1e-4):
        g = np.zeros_like(p)
        for c in range(2):
            q = p.copy()
            q[:, c] = q[:, c] + h * (0.5 * (q[:, c].min() + (q[:, c].max() + L)))
            g[:, c] = (_volume_exact(q) - best_v) / h
        return g

    step = 1e-3
    for _it in range(30000):
        if time.time() - start > DEADLINE:
            break
        g = grad(pos)
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        # try a few steps with backtracking
        accepted = False
        for _ in range(12):
            cand = pos - step * g
            v = _volume_exact(cand)
            if np.isfinite(v) and v < best_v - 1e-12:
                pos = cand
                best_v = v
                best = pos.copy()
                accepted = True
                break
            step *= 0.5
        if not accepted:
            step *= 0.5
            if step < 1e-7:
                break

        # ---- global box-shrink (the key lever) ----
        accepted = False
        for alpha in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7):
            cand = shrink(pos, alpha)
            v = _volume_exact(cand)
            if np.isfinite(v) and v < best_v - 1e-12:
                pos = cand
                best_v = v
                best = pos.copy()
                accepted = True
                break
        if not accepted:
            # gradient-based shrink
            sg = shrink_grad(pos)
            sn = np.linalg.norm(sg)
            if sn > 1e-12:
                sstep = 1e-3
                for _ in range(10):
                    cand = pos - sstep * sg
                    v = _volume_exact(cand)
                    if np.isfinite(v) and v < best_v - 1e-12:
                        pos = cand
                        best_v = v
                        best = pos.copy()
                        break
                    sstep *= 0.5
    return best, best_v


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    pos0 = _base_positions()
    best, best_v = _optimize(pos0, start)

    # safety: ensure finite
    if not np.isfinite(best_v):
        return pos0
    # final deadline guard
    if time.time() - start > DEADLINE:
        return best
    return best
