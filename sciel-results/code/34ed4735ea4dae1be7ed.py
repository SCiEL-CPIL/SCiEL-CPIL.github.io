import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
T0 = time.time()
DEADLINE = 110.0          # safety margin before the 120s kill
TIME_PER_START = 10.0     # wall-clock budget per optimization start
BEST = {"side": 40.0, "inner": None}

# canonical regular hexagon of side 1, center at origin, vertex 0 at angle 0
CANON = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
CANON = CANON.reshape(-1, 2)
# outward unit edge normals (directions used as support directions)
def _edge_normals():
    out = []
    for i in range(6):
        a = CANON[i]
        b = CANON[(i + 1) % 6]
        e = b - a
        e = e / np.linalg.norm(e)
        n = np.array([e[1], -e[0]])
        n = n / np.linalg.norm(n)
        if n @ a < 0:
            n = -n
        out.append(n)
    return np.array(out)
NORMS = _edge_normals()


def _rot(v, a):
    c = math.cos(a)
    s = math.sin(a)
    return v * c + v[:, None] * np.cross(v[:, None], CANON[0:1]) * 0  # placeholder
def _rot2(v, a):
    c = math.cos(a)
    s = math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return v @ R.T


def _mkvex(za, zb):
    # Minkowski-difference vertices for two hexagons with local rotations za, zb
    A = _rot2(CANON, za)
    B = _rot2(CANON, zb)
    C = A[:, None, :] - B[None, :, :]          # (6,6,2)
    C = C.reshape(-1, 2)
    ang = np.arctan2(C[:, 1], C[:, 0])
    order = np.argsort(ang)
    C = C[order]
    c = C.mean(axis=0)
    C = C - c
    C = C / np.linalg.norm(C, axis=1)[:, None]
    return C


def _mkvex_cache(za, zb):
    key = (round(za, 6), round(zb, 6))
    return _mkvex(za, zb)


def _mkvexs(zas, zbs):
    return [_mkvex(za, zb) for za, zb in zip(zas, zbs)]


def _hmax(C, d):
    # max support of vertex set C along directions d (rows)
    return np.max(C @ d.T, axis=1)


def _support(C, d):
    return np.max(C @ d.T)


def _objective(x, za, zb, c1, c2, side):
    # x = [dx, dy] relative displacement of hexagon 2 from hexagon 1
    dx, dy = x
    d = np.array([dx, dy])
    h = _hmax(C, d)
    # violation when support < 1
    return max(0.0, 1.0 - h)


def _make_constraints(zas, zbs, c1, c2, side):
    # precompute Minkowski vertex sets per pair
    mk = {}
    for i in range(N):
        for j in range(i + 1, N):
            mk[(i, j)] = _mkvex(zas[i], zas[j])

    def gfun(x):
        # x = [x0,y0,th0, x1,y1,th1, ...]  (6N)
        pos = x[0:2 * N].reshape(N, 2)
        # th = x[2*N:3*N]
        vals = np.zeros(N * (N - 1) // 2)
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                C = mk[(i, j)]
                d = pos[j] - pos[i]
                h = np.max(C @ d)
                vals[k] = h - 1.0   # >=0 required
                k += 1
        return vals

    cons = [{'type': 'ineq', 'fun': gfun}]
    return cons


def _contains(pos, th, side):
    # all inner hexagons inside outer regular hexagon of side `side`
    # outer support direction set = NORMS (orientation 0)
    A = _rot2(CANON, 0.0)
    for i in range(N):
        B = _rot2(CANON, th[i])
        C = A[:, None, :] - B[None, :, :]
        C = C.reshape(-1, 2)
        ang = np.arctan2(C[:, 1], C[:, 0])
        C = C[np.argsort(ang)]
        C = C - C.mean(axis=0)
        C = C / np.linalg.norm(C, axis=1)[:, None]
        d = pos[i]
        h = np.max(C @ d)
        if h > side:
            return False
    return True


def _valid(pos, th, side):
    if not _contains(pos, th, side):
        return False
    mk = {}
    for i in range(N):
        for j in range(i + 1, N):
            C = _mkvex(th[i], th[j])
            d = pos[j] - pos[i]
            if np.max(C @ d) < 1.0 - 1e-9:
                return False
    return True


def _inner_rows(pos, th):
    return [[float(pos[i, 0]), float(pos[i, 1]), float(np.degrees(th[i]))] for i in range(N)]


def _starts():
    starts = []
    # 1) hexagonal lattice
    pos = []
    for r in range(-1, 2):
        for c in range(-1, 2):
            x = 3.0 * c + (1.5 if r % 2 else 0.0)
            y = math.sqrt(3) * r
            pos.append([x, y])
    pos = np.array(pos[:N])
    th = np.zeros(N)
    starts.append((pos, th))
    # 2) circular
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rad = 3.2
    pos = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    th = np.zeros(N)
    starts.append((pos, th))
    # 3) random
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, size=(N, 2))
    th = rng.uniform(0, 2 * np.pi, size=N)
    starts.append((pos, th))
    # 4) random lattice-ish
    rng = np.random.default_rng(7)
    pos = rng.uniform(-4, 4, size=(N, 2))
    th = np.zeros(N)
    starts.append((pos, th))
    return starts


def construct():
    global BEST
    side = 40.0
    for (pos0, th0) in _starts():
        if time.time() > DEADLINE - 5:
            break
        x0 = np.concatenate([pos0.reshape(-1), th0])
        cons = _make_constraints(th0, th0, 0, 0, side)
        # we will optimize positions+angles for a fixed side; then reduce side
        # Strategy: fix a candidate side, check feasibility; binary search side.
        lo, hi = 3.0, 40.0
        # first ensure hi is feasible with a quick feasible push (skip if not)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if time.time() > DEADLINE - 5:
                break
            feas = _try_side(mid, x0, th0)
            if feas is None:
                break
            if feas:
                hi = mid
            else:
                lo = mid
        if hi < BEST["side"]:
            # recompute best inner at side hi
            inner, ok = _solve_side(hi, x0, th0)
            if ok and _valid(np.array([r[0] for r in inner]),
                             np.array([math.radians(r[2]) for r in inner]), hi):
                BEST["side"] = hi
                BEST["inner"] = inner
    inner = BEST["inner"] if BEST["inner"] is not None else [[0.0, 0.0, 0.0] for _ in range(N)]
    return inner, (0.0, 0.0), float(BEST["side"]), 0.0


def _try_side(side, x0, th0):
    cons = _make_constraints(th0, th0, 0, 0, side)
    def obj(x):
        pos = x[0:2 * N].reshape(N, 2)
        th = x[2 * N:3 * N]
        if not _contains(pos, th, side):
            return 1e6
        mk = {}
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                C = _mkvex(th[i], th[j])
                d = pos[j] - pos[i]
                pen += max(0.0, 1.0 - np.max(C @ d))
        return pen
    try:
        res = minimize(obj, x0, method='SLSQP',
                       options={'maxiter': 60, 'ftol': 1e-5})
    except Exception:
        return None
    x = res.x
    pos = x[0:2 * N].reshape(N, 2)
    th = x[2 * N:3 * N]
    if not _contains(pos, th, side):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            C = _mkvex(th[i], th[j])
            d = pos[j] - pos[i]
            if np.max(C @ d) < 1.0 - 1e-7:
                return False
    return True


def _solve_side(side, x0, th0):
    def obj(x):
        pos = x[0:2 * N].reshape(N, 2)
        th = x[2 * N:3 * N]
        if not _contains(pos, th, side):
            return 1e6
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                C = _mkvex(th[i], th[j])
                d = pos[j] - pos[i]
                pen += max(0.0, 1.0 - np.max(C @ d))
        return pen
    try:
        res = minimize(obj, x0, method='SLSQP', options={'maxiter': 80, 'ftol': 1e-6})
    except Exception:
        return None, False
    x = res.x
    pos = x[0:2 * N].reshape(N, 2)
    th = x[2 * N:3 * N]
    inner = _inner_rows(pos, th)
    return inner, True


if __name__ == "__main__":
    r = construct()
    print(r[2])
