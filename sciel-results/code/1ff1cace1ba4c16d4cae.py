# EVOLVE-BLOCK-START
import numpy as np
import time

_N = 64
_A = (np.arange(_N) + 1) / _N
_Y_FINE = (np.arange(20000) + 0.5) / 20000
_W_FINE = (1 - _Y_FINE) / _N

# Coarse grid for fast search
_NH = 4000
_Y = (np.arange(_NH) + 0.5) / _NH
_W = (1 - _Y) / _N


def _area_coarse(x):
    """Union area on the coarse grid (cheap)."""
    L = x[:, None] + _A[:, None] * _Y[None, :]
    Ls = np.sort(L, axis=0)
    d = np.diff(Ls, axis=0)
    union = _W[None, :] + np.sum(np.minimum(d, _W[None, :]), axis=0)
    return float(union.mean())


def _area_fine(x):
    """Union area on the fine (evaluator) grid (exact metric)."""
    L = x[:, None] + _A[:, None] * _Y_FINE[None, :]
    Ls = np.sort(L, axis=0)
    d = np.diff(Ls, axis=0)
    union = _W_FINE[None, :] + np.sum(np.minimum(d, _W_FINE[None, :]), axis=0)
    return float(union.mean())


def _keich():
    n = _N
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    pos = np.empty(n)
    for i in range(n):
        b = bin(i)[2:]
        eps = np.array(list(b.zfill(k)), dtype=np.float64)
        pos[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return pos


def _golden_min(f, a, b, iters=40):
    """Minimize unimodal f on [a,b] via golden-section + parabolic polish."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    # parabolic polish
    x1, x2, x3 = a, (a + b) / 2.0, b
    f1, f2, f3 = f(x1), f(x2), f(x3)
    for _ in range(10):
        denom = (f1 - f3) * (x2 - x1) + (f1 - f2) * (x3 - x2)
        if abs(denom) < 1e-30:
            break
        xc = 0.5 * (x1 + x2) + 0.5 * (f1 - f3) * (x2 - x1) ** 2 / denom \
            - 0.5 * (f1 - f2) * (x3 - x2) ** 2 / denom
        if not (x1 < xc < x2):
            break
        fc = f(xc)
        if fc < f2:
            x1, f1 = x2, f2
            x2, f2 = xc, fc
        else:
            x3, f3 = xc, fc
    # pick best of the three
    cands = [(x1, f1), (x2, f2), (x3, f3)]
    cands.sort(key=lambda t: t[1])
    return cands[0][0]


def _coord_descent(x, iters, deadline):
    """Coordinate descent with golden-section line search on coarse area."""
    x = x.copy()
    cur = _area_coarse(x)
    for it in range(iters):
        if time.time() > deadline:
            break
        improved = False
        for i in range(_N):
            if time.time() > deadline:
                break
            old = x[i]

            def fi(v):
                x[i] = v
                return _area_coarse(x)

            # bracket around current
            lo, hi = old - 0.05, old + 0.05
            fl, fm, fh = fi(lo), fi(old), fi(hi)
            # expand if needed
            for _ in range(20):
                if fl < fm and fm < fh:
                    break
                lo = lo - (hi - lo)
                hi = hi + (hi - lo)
                fl = fi(lo)
                fh = fi(hi)
                if fl < fm and fm < fh:
                    break
            x[i] = _golden_min(fi, lo, hi, iters=35)
            nf = _area_coarse(x)
            if nf < cur - 1e-14:
                cur = nf
                improved = True
        if not improved:
            break
    return x, cur


def _subgrad_polish(x, deadline):
    """Piecewise-linear subgradient sweep: move each coordinate to a candidate
    breakpoint (where some pair of L-values equal) to reduce coarse area."""
    x = x.copy()
    cur = _area_coarse(x)
    # candidate deltas: breakpoints where x_i + a_i y = x_j + a_j y  =>  x_i - x_j = (a_j - a_i) y
    # y in [0,1]; collect a spread of y
    ys = np.linspace(0.0, 1.0, 41)
    for i in range(_N):
        if time.time() > deadline:
            break
        base = x[i]
        best_x, best_a = base, cur
        for j in range(_N):
            if j == i:
                continue
            for y in ys:
                cand = base + (_A[j] - _A[i]) * y
                x[i] = cand
                a = _area_coarse(x)
                if a < best_a - 1e-15:
                    best_a = a
                    best_x = cand
        x[i] = best_x
        if best_a < cur - 1e-15:
            cur = best_a
    return x, cur


def kakeya_needle_2d_positions64() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0

    best_x = _keich()
    best_fine = _area_fine(best_x)

    # A few candidate starts
    starts = [_keich()]
    rng = np.random.default_rng(0)
    for s in range(6):
        starts.append(_keich() + rng.normal(0, 0.05, _N))
    for s in range(4):
        starts.append(rng.uniform(-0.5, 0.5, _N))

    for st in starts:
        if time.time() > deadline:
            break
        x, _ = _coord_descent(st, iters=60, deadline=deadline)
        x, _ = _subgrad_polish(x, deadline)
        # short refine
        x, _ = _coord_descent(x, iters=20, deadline=deadline)
        af = _area_fine(x)
        if af < best_fine:
            best_fine = af
            best_x = x.copy()

    # Final high-precision polish
    if time.time() < deadline:
        x, _ = _subgrad_polish(best_x, deadline)
        x, _ = _coord_descent(x, iters=30, deadline=deadline)
        af = _area_fine(x)
        if af < best_fine:
            best_x = x

    out = np.asarray(best_x, dtype=np.float64).reshape(64)
    return out


# EVOLVE-BLOCK-END
