import numpy as np
import time

N = 12
DEADLINE = 145.0
_T0 = time.time()

# Precompute unit hexagon geometry (flat-top, vertex at angle 0)
_HX = np.cos(np.radians([0, 60, 120, 180, 240, 300]))
_HY = np.sin(np.radians([0, 60, 120, 180, 240, 300]))
_HEX = np.column_stack((_HX, _HY))                      # (6,2)
_E = np.roll(_HEX, -1, axis=0) - _HEX                  # edge vectors (6,2)
_NORM = np.linalg.norm(_E, axis=1)
_N = np.column_stack((-_E[:, 1], _E[:, 0])) / _NORM[:, None]   # outward normals (6,2)
# support function of unit hexagon along each of its own 6 normals
_SUP = np.abs(_HEX @ _N.T).max(axis=1)                 # (6,)

def _elapsed():
    return time.time() - _T0

def _time_left():
    return DEADLINE - _elapsed()

def _rot(R, deg):
    """2D rotation matrix for a given angle (degrees)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])

def _outer_side(pts, rot_deg):
    """Minimal side of a centered regular hexagon (rotation rot_deg) enclosing pts.
    pts: (M,2). Returns scalar side length."""
    Nout = _rot(_N, rot_deg)                            # (6,2)
    sup = np.abs(pts @ Nout.T).max(axis=0)              # (6,)
    return (2.0 / np.sqrt(3.0)) * sup.max()

def _batch_vertices(xys, rots_deg):
    """xys: (N,2), rots_deg: (N,). Returns (N,6,2)."""
    R = np.array([_rot(np.eye(2), r) for r in rots_deg])      # (N,2,2)
    v = np.einsum('nij,nj->ni', R, _HEX) + xys[:, None, :]    # (N,6,2)
    return v

def _min_separation(v):
    """v: (N,6,2). Return (sep, overlap) where sep>=0 is the true min gap between
    distinct hexagons (0 if touching/overlapping) and overlap>0 is total penetration."""
    Nn = len(v)
    sep = float('inf')
    overlap = 0.0
    for i in range(Nn):
        for j in range(i + 1, Nn):
            d = v[j] - v[i]                    # (6,2)
            # axes: 6 normals of hex i, 6 normals of hex j
            proj_i = d @ _N.T                  # (6,)
            proj_j = d @ _N[j].T               # (6,)
            for a in range(6):
                ov = (_SUP[a] + _SUP[a]) - abs(proj_i[a])
                if ov > 0:
                    overlap += ov
                else:
                    s = -ov
                    if s < sep:
                        sep = s
            for a in range(6):
                ov = (_SUP[a] + _SUP[a]) - abs(proj_j[a])
                if ov > 0:
                    overlap += ov
                else:
                    s = -ov
                    if s < sep:
                        sep = s
    return sep, overlap

def _objective(params):
    if _time_left() < 1.0:
        return 1e9
    xys = params[:2 * N].reshape(N, 2)
    rots = params[2 * N:3 * N]
    outer = params[3 * N]
    v = _batch_vertices(xys, rots)
    sep, overlap = _min_separation(v)
    pts = v.reshape(-1, 2)
    L = _outer_side(pts, outer)
    if overlap > 1e-9:
        return L + 100.0 * overlap
    if sep < 1e-7:
        return L + 100.0 * (1e-7 - sep)
    return L

def _eval(params):
    xys = params[:2 * N].reshape(N, 2)
    rots = params[2 * N:3 * N]
    outer = params[3 * N]
    v = _batch_vertices(xys, rots)
    sep, overlap = _min_separation(v)
    if overlap > 1e-9 or sep < 1e-7:
        return None
    pts = v.reshape(-1, 2)
    return _outer_side(pts, outer)

def _make_ring():
    """1 center + 6 ring (dist sqrt3) + 5 outer (dist 2*sqrt3)."""
    s3 = np.sqrt(3.0)
    pts = [[0.0, 0.0]]
    for k in range(6):
        a = np.radians(60 * k)
        pts.append([s3 * np.cos(a), s3 * np.sin(a)])
    for k in range(5):
        a = np.radians(60 * k + 30)
        pts.append([2 * s3 * np.cos(a), 2 * s3 * np.sin(a)])
    return np.array(pts, dtype=float)

def _make_grid(rows, cols):
    """Offset hex grid, flat-top unit hexagons, centered."""
    s3 = np.sqrt(3.0)
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5 * (r % 2)) * s3
            y = r * 1.5
            pts.append([x, y])
    pts = np.array(pts, dtype=float)
    pts = pts[:N]
    pts -= pts.mean(axis=0)
    return pts

def _center(pts):
    """Recenter a (N,2) point set to the origin."""
    return pts - pts.mean(axis=0)

def construct():
    start = time.time()

    def tl():
        return DEADLINE - (time.time() - start)

    rng = np.random.default_rng(20240601)

    # --- Build diverse initial population ---
    inits = []
    inits.append(_make_ring())
    inits.append(_make_grid(4, 3))
    inits.append(_make_grid(3, 4))
    inits.append(_make_grid(2, 6))
    inits.append(_make_grid(6, 2))
    inits.append(_make_grid(3, 3) + rng.normal(0, 0.1, (9, 2)))
    # random scatter
    for _ in range(6):
        inits.append(rng.normal(0, 2.5, (N, 2)))

    pop = []
    for pts in inits:
        rots = np.zeros(N)
        outer = 0.0
        p = np.concatenate((pts.reshape(-1), rots, [outer]))
        pop.append(p)
    pop = np.array(pop, dtype=float)

    best = None
    best_val = float('inf')
    for p in pop:
        val = _eval(p)
        if val is not None and val < best_val:
            best_val = val
            best = p.copy()

    # --- Differential evolution (global search) ---
    bounds_lo = np.concatenate((np.full(2 * N, -15.0), np.full(N, -90.0), [-180.0]))
    bounds_hi = np.concatenate((np.full(2 * N, 15.0), np.full(N, 90.0), [180.0]))
    F = 0.7
    CR = 0.8
    pop_size = pop.shape[0]
    # keep population at least 5
    while pop_size < 5:
        pop = np.vstack([pop, pop[0]])
        pop_size += 1

    while tl() > 12.0:
        new_pop = [pop[0].copy()]  # keep best of current generation
        for i in range(1, pop_size):
            if tl() < 2.0:
                break
            # pick three distinct parents
            while True:
                a, b, c = rng.integers(0, pop_size, 3)
                if len(set((a, b, c))) == 3 and a != i:
                    break
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, bounds_lo, bounds_hi)
            trial = np.empty_like(mutant)
            mask = rng.random(mutant.shape) < CR
            if not mask.any():
                mask[rng.integers(mutant.shape)] = True
            trial[mask] = mutant[mask]
            trial[~mask] = pop[i][~mask]
            # evaluate both, keep the better (penalty-based)
            ft = _objective(trial)
            fp = _objective(pop[i])
            if ft <= fp:
                new_pop.append(trial)
            else:
                new_pop.append(pop[i].copy())
        pop = np.array(new_pop)
        # track best valid
        for p in pop:
            val = _eval(p)
            if val is not None and val < best_val:
                best_val = val
                best = p.copy()

    # --- Local refinement of the best valid solution ---
    if best is not None and tl() > 8.0:
        from scipy.optimize import minimize
        cur = best.copy()
        cur_val = best_val
        for _ in range(2):
            if tl() < 6.0:
                break
            try:
                res = minimize(_objective, cur, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-6,
                                        'fatol': 1e-8, 'adaptive': True})
                val = _eval(res.x)
                if val is not None and val < cur_val:
                    cur_val = val
                    cur = res.x.copy()
            except Exception:
                break
        best = cur
        best_val = cur_val

    # --- Fallback if nothing valid found ---
    if best is None:
        pts = _make_ring()
        rots = np.zeros(N)
        outer = 0.0
        best = np.concatenate((pts.reshape(-1), rots, [outer]))
        best_val = _outer_side(pts.reshape(-1, 2), 0.0)

    xys = best[:2 * N].reshape(N, 2)
    rots = best[2 * N:3 * N]
    outer = best[3 * N]

    inner = [[float(xys[i, 0]), float(xys[i, 1]), float(rots[i])] for i in range(N)]
    center = [0.0, 0.0]
    side = float(best_val)
    angle = float(outer)
    return inner, center, side, angle
