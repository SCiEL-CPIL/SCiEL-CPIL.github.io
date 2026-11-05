"""Photonic-crystal unit-cell designer. Combines a diverse candidate generator
with a fast TM plane-wave gap evaluator and local refinement under a hard
deadline guard."""
import time
import numpy as np
from numpy.linalg import eigh

_NPW = 17          # plane-wave radius (Gx,Gy with Gx^2+Gy^2 <= NPW^2)
_EPS_HIGH = 12.0
_EPS_LOW = 1.0

# ---- irreducible Brillouin zone of the square lattice (fractions of 2pi) ----
# Path Gamma -> X -> M -> Gamma covers the whole irreducible BZ.
_KPTS = []
for t in np.linspace(0.0, 0.5, 6):      # Gamma -> X  (kx,0)
    _KPTS.append((t, 0.0))
for t in np.linspace(0.0, 0.5, 6):      # X -> M     (0.5,ky)
    _KPTS.append((0.5, t))
for t in np.linspace(0.5, 0.0, 5):      # M -> Gamma (t,t)
    _KPTS.append((t, t))
_KPTS = np.array(_KPTS, dtype=float)


def _build_pw():
    """Precompute the plane-wave basis and the inverse-permittivity kernel
    (as a function of the pixel map)."""
    G = []
    for Gx in range(-_NPW, _NPW + 1):
        for Gy in range(-_NPW, _NPW + 1):
            if Gx * Gx + Gy * Gy <= _NPW * _NPW:
                G.append((Gx, Gy))
    G = np.array(G, dtype=float)
    n = len(G)
    # difference table of indices (for the kernel lookup)
    d = G[:, None, :] - G[None, :, :]          # (n, n, 2)
    return G, n, d


_G, _NPWN, _D = _build_pw()
_N = 32
# (k+G)^2 table per k-point: (n_k, n_pw)
_KG2 = np.array([np.sum((_KPTS[:, None, :] + _G[None, :, :]) ** 2, axis=2)
                 for _ in range(0)])  # placeholder, filled lazily below
_KG2 = None


def _kg2():
    global _KG2
    if _KG2 is None:
        kg = _KPTS[:, None, :] + _G[None, :, :]
        _KG2 = np.sum(kg * kg, axis=2)          # (n_k, n_pw)
    return _KG2


def _kernel(cell):
    """Inverse-permittivity Fourier coefficients, flattened by the diff table."""
    eps = _EPS_LOW + (_EPS_HIGH - _EPS_LOW) * np.array(cell, dtype=float)
    inv = 1.0 / eps
    F = np.fft.fft2(inv) / (_N * _N)
    # index into F by the wrapped difference (mod N)
    idx = np.mod(_D, _N)                        # (n, n, 2)
    return F[idx[:, :, 0], idx[:, :, 1]]        # (n, n) complex


def _gap(cell, n_k=None):
    """Estimate the 4/5 TM gap-midgap ratio (higher = better)."""
    if n_k is None:
        n_k = len(_KPTS)
    kg2 = _kg2()[:n_k]
    F = _kernel(cell)
    max_w4 = 0.0
    min_w5 = np.inf
    n = _NPWN
    for kk in range(n_k):
        M = kg2[kk][None, :, None] * F          # broadcast (n,1,1)*(n,n) -> (n,n)
        M = 0.5 * (M + M.conj().T)
        try:
            ev = eigh(M, eigvals_only=True)
        except np.linalg.LinAlgError:
            continue
        ev = np.sqrt(np.maximum(ev, 0.0))
        if ev.size >= 5:
            w4 = float(ev[3])
            w5 = float(ev[4])
            if w4 > max_w4:
                max_w4 = w4
            if w5 < min_w5:
                min_w5 = w5
    if not np.isfinite(min_w5) or max_w4 <= 0.0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _symmetrize(c):
    """Average over the 4-fold symmetry group and threshold to 0/1."""
    c = np.array(c, dtype=float)
    avg = (c + c.T + c[::-1, :] + c[:, ::-1]) / 4.0
    return (avg > 0.5).astype(int)


def _cell_from_quad(quad):
    """Build a full symmetric 32x32 cell from a 16x16 fundamental quadrant."""
    q = np.array(quad, dtype=int)
    full = np.block([[q, q[:, ::-1]], [q[::-1, :], q[::-1, ::-1]]])
    return full.tolist()


def _quad_from_cell(cell):
    c = np.array(cell, dtype=int)
    return c[:_N // 2, :_N // 2]


# ---- candidate generators (all return symmetric 32x32 cells) ----
def _gen_rod(r):
    cell = np.zeros((_N, _N), dtype=int)
    x = (np.arange(_N) + 0.5) / _N - 0.5
    xx, yy = np.meshgrid(x, x)
    cell[(xx * xx + yy * yy) < r * r] = 1
    return cell.tolist()


def _gen_hole(r):
    cell = np.ones((_N, _N), dtype=int)
    x = (np.arange(_N) + 0.5) / _N - 0.5
    xx, yy = np.meshgrid(x, x)
    cell[(xx * xx + yy * yy) < r * r] = 0
    return cell.tolist()


def _gen_ring(r_in, r_out):
    cell = np.zeros((_N, _N), dtype=int)
    x = (np.arange(_N) + 0.5) / _N - 0.5
    xx, yy = np.meshgrid(x, x)
    d2 = xx * xx + yy * yy
    cell[(d2 < r_out * r_out) & (d2 > r_in * r_in)] = 1
    return cell.tolist()


def _gen_square(r):
    cell = np.zeros((_N, _N), dtype=int)
    x = (np.arange(_N) + 0.5) / _N - 0.5
    xx, yy = np.meshgrid(x, x)
    cell[(np.abs(xx) < r) & (np.abs(yy) < r)] = 1
    return cell.tolist()


def _gen_cross(w, L):
    cell = np.zeros((_N, _N), dtype=int)
    x = (np.arange(_N) + 0.5) / _N - 0.5
    xx, yy = np.meshgrid(x, x)
    cell[((np.abs(xx) < w / 2) & (np.abs(yy) < L / 2)) |
         ((np.abs(yy) < w / 2) & (np.abs(xx) < L / 2))] = 1
    return cell.tolist()


def _gen_diagonal(w, L):
    cell = np.zeros((_N, _N), dtype=int)
    x = (np.arange(_N) + 0.5) / _N - 0.5
    xx, yy = np.meshgrid(x, x)
    u = (xx + yy) / np.sqrt(2.0)
    v = (xx - yy) / np.sqrt(2.0)
    cell[(np.abs(u) < w / 2) & (np.abs(v) < L / 2)] = 1
    return cell.tolist()


def _gen_random_quad(density, seed):
    rng = np.random.RandomState(seed)
    quad = (rng.rand(_N // 2, _N // 2) < density).astype(int)
    return _cell_from_quad(quad)


def _mutate_quad(quad, rate, seed):
    rng = np.random.RandomState(seed)
    q = np.array(quad, dtype=int).copy()
    h, w = q.shape
    flip = rng.rand(h, w) < rate
    q[flip] = 1 - q[flip]
    return q


def design(problem):
    t0 = time.time()
    deadline = t0 + 210.0        # hard wall-clock guard (evaluator kills at 230s)
    size = problem["pixels"]
    assert size == _N

    def remaining():
        return deadline - time.time()

    def fast_gap(cell):
        return _gap(cell, n_k=4)   # coarse: only the high-symmetry k-points

    def full_gap(cell):
        return _gap(cell)

    # ---------- Stage 1: diverse candidate pool ----------
    pool = []
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        pool.append(_gen_rod(r))
    for r in [0.3, 0.35, 0.4, 0.45, 0.5]:
        pool.append(_gen_hole(r))
    for (ri, ro) in [(0.15, 0.3), (0.2, 0.35), (0.1, 0.28), (0.25, 0.42),
                     (0.18, 0.4), (0.3, 0.45)]:
        pool.append(_gen_ring(ri, ro))
    for r in [0.15, 0.2, 0.25, 0.3]:
        pool.append(_gen_square(r))
    for (w, L) in [(0.1, 0.5), (0.15, 0.55), (0.12, 0.45), (0.2, 0.6)]:
        pool.append(_gen_cross(w, L))
        pool.append(_gen_diagonal(w, L))
    # random quads across a range of densities (higher-order gaps need variety)
    for s in range(24):
        density = 0.15 + 0.03 * s
        pool.append(_gen_random_quad(density, seed=1000 + s))

    # coarse screen
    best_cell = None
    best_score = -1.0
    for cell in pool:
        if remaining() < 40.0:
            break
        try:
            sc = fast_gap(cell)
        except Exception:
            continue
        if sc > best_score:
            best_score = sc
            best_cell = cell

    if best_cell is None:
        best_cell = _gen_rod(0.2)
        best_score = fast_gap(best_cell)

    # refine the current best with a full-k evaluation
    full_best = full_gap(best_cell)
    if full_best > best_score:
        best_score = full_best

    # ---------- Stage 2: local refinement around the best ----------
    cur_quad = _quad_from_cell(best_cell)
    cur_score = best_score
    step = 0
    while remaining() > 50.0:
        step += 1
        rate = 0.05 if step < 12 else 0.02
        mutated = _mutate_quad(cur_quad, rate, seed=2000 + step)
        cand = _cell_from_quad(mutated)
        try:
            sc = fast_gap(cand)
        except Exception:
            continue
        if sc > cur_score - 0.02:           # accept near-ties to explore
            fs = full_gap(cand)
            if fs > cur_score:
                cur_score = fs
                cur_quad = mutated
                best_cell = cand
                best_score = fs

    # ---------- Stage 3: a few fresh random restarts if time remains ----------
    restart = 0
    while remaining() > 25.0:
        restart += 1
        density = 0.15 + 0.02 * (restart % 20)
        cand = _gen_random_quad(density, seed=5000 + restart)
        try:
            sc = fast_gap(cand)
        except Exception:
            continue
        if sc > best_score - 0.01:
            fs = full_gap(cand)
            if fs > best_score:
                best_score = fs
                best_cell = cand

    # final safety: ensure a valid symmetric 0/1 list
    out = _symmetrize(best_cell)
    return out.tolist()
