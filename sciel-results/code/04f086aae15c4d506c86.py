"""Circle packing in unit square: hex-lattice init + penalty L-BFGS-B + local fix."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

_DEADLINE = 150.0          # evaluator wall-clock kill
_MARGIN = 5.0              # safety margin before returning
_START = time.time()


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _hex_lattice(n, margin=0.02):
    """Dense hexagonal packing of n circles, then scale to fit in [margin,1-margin]^2."""
    # hex lattice: rows offset
    pts = []
    row = 0
    while len(pts) < n:
        y = row * (np.sqrt(3) / 2)
        xoff = (row % 2) * 0.5
        col = 0
        while len(pts) < n:
            pts.append((col + xoff, y))
            col += 1
        row += 1
    pts = np.array(pts[:n])
    # normalize into square
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    span = (pmax - pmin).max()
    if span <= 0:
        span = 1.0
    scale = (1.0 - 2 * margin) / span
    pts = (pts - pmin) * scale + margin
    # initial radius estimate: half the min nearest-neighbour distance
    dmin = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d[i] = np.inf
        dmin = min(dmin, d.min())
    r0 = max(0.005, 0.5 * dmin * 0.98)
    return pts, r0


def _make_objective(n, lam):
    """Penalty objective: -sum(r) + lam*(overlap + boundary violations)."""
    def obj(v):
        p = v[:2 * n].reshape(n, 2)
        r = np.abs(v[2 * n:]) + 1e-9
        # boundary violations: circle must stay inside [0,1]^2
        bnd = np.maximum(0, p[:, 0] - r) + np.maximum(0, (1 - p[:, 0]) - r) \
            + np.maximum(0, p[:, 1] - r) + np.maximum(0, (1 - p[:, 1]) - r)
        # pairwise overlap
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        need = r[:, None] + r[None, :]
        ov = np.maximum(0, need - d)
        # zero diagonal
        ov[np.diag_indices(n)] = 0.0
        return -r.sum() + lam * (ov.sum() + bnd.sum())
    return obj


def _fix_local(p, r, iters=2000, step=0.002):
    """Greedy local repair: shrink/shift to remove overlaps and boundary violations."""
    p = p.copy()
    r = r.copy()
    rng = np.random.default_rng(0)
    for it in range(iters):
        # boundary clamp
        r = np.minimum(r, np.minimum(p[:, 0], 1 - p[:, 0]))
        r = np.minimum(r, np.minimum(p[:, 1], 1 - p[:, 1]))
        r = np.maximum(r, 1e-6)
        # check pairwise
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        need = r[:, None] + r[None, :]
        ov = np.maximum(0, need - d)
        ov[np.diag_indices(n)] = 0.0
        if ov.max() < 1e-7:
            break
        # find worst pair
        ii, jj = np.unravel_index(np.argmax(ov), ov.shape)
        if ii == jj:
            break
        excess = ov[ii, jj]
        # shrink the larger radius a bit and push apart
        # push apart along connecting vector
        vec = p[jj] - p[ii]
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            vec = rng.normal(size=2)
            norm = np.linalg.norm(vec)
        unit = vec / norm
        half = 0.5 * excess
        p[ii] -= half * unit
        p[jj] += half * unit
        # also shrink the bigger one slightly
        big = ii if r[ii] >= r[jj] else jj
        r[big] = max(1e-6, r[big] - 0.5 * excess)
    return p, r


def _valid(p, r, tol=1e-6):
    # boundary
    if np.any(p - r < -tol) or np.any(p + r > 1 + tol):
        return False
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    need = r[:, None] + r[None, :]
    if np.any(need - d > tol):
        return False
    return True


def construct():
    t0 = time.time()
    # initial hex lattice
    p0, r0 = _hex_lattice(N)
    x0 = np.concatenate([p0.ravel(), np.full(N, r0)])

    # bounds: positions in [0,1], radii in [1e-4, 0.5]
    lb = np.concatenate([np.zeros(2 * N), np.full(N, 1e-4)])
    ub = np.concatenate([np.ones(2 * N), np.full(N, 0.5)])

    best_x = x0.copy()
    best_val = -np.inf
    best_p, best_r = p0.copy(), r0.copy()

    lam = 1.0
    for it in range(40):
        if _remaining() < 2.0:
            break
        obj = _make_objective(N, lam)
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                           options={'maxiter': 200, 'maxfun': 2000, 'gtol': 1e-8})
        except Exception:
            break
        x0 = res.x
        p = x0[:2 * N].reshape(N, 2)
        r = np.abs(x0[2 * N:])
        # local repair
        p, r = _fix_local(p, r, iters=1500)
        val = r.sum()
        if val > best_val:
            best_val = val
            best_p, best_r = p.copy(), r.copy()
            best_x = np.concatenate([p.ravel(), r])
        # increase penalty as overlaps shrink
        lam *= 1.6
        if _remaining() < 3.0:
            break

    p, r = best_p, best_r
    # final safety repair
    p, r = _fix_local(p, r, iters=3000)
    # final validity clamp
    r = np.minimum(r, np.minimum(p[:, 0], 1 - p[:, 0]))
    r = np.minimum(r, np.minimum(p[:, 1], 1 - p[:, 1]))
    r = np.maximum(r, 1e-6)
    out = np.column_stack([p, r])
    return out
