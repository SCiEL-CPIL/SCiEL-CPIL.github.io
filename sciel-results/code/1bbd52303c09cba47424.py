"""Design a symmetric 32x32 photonic-crystal unit cell to maximize the TM band-4/5 gap.

Structural idea (from the best-scoring inspiration): start from a 4-rod geometry
with rods at (+/-0.25, +/-0.25). Refinement machinery: a fast 289-plane-wave
Galerkin evaluator (scipy sparse eigsh) plus a pixel-level orbit-flip local
search that explores the full binary design space around that seed.
"""
import time
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0
_EPS_INV_MEAN = 0.5 * (1.0 / _EPS_HIGH + 1.0 / _EPS_LOW)
_EPS_INV_AMP = 0.5 * (1.0 / _EPS_HIGH - 1.0 / _EPS_LOW)

# Plane-wave basis: 289 = 17x17 grid of G vectors
_G_RANGE = 8
_GV = np.array([(m, n) for m in range(-_G_RANGE, _G_RANGE + 1)
                for n in range(-_G_RANGE, _G_RANGE + 1)], dtype=np.float64)
_NPW = _GV.shape[0]  # 289

# Fourier-coupling table: _COUP[i, j] = index of (G_i - G_j) in the 17x17 grid,
# or -1 if that difference falls outside the retained G set.
_gx = _GV[:, 0]
_gy = _GV[:, 1]
_dx = gx[:, None] - gx[None, :]
_dy = gy[:, None] - gy[None, :]
_in = (np.abs(_dx) <= _G_RANGE) & (np.abs(_dy) <= _G_RANGE)
_COUP = np.where(_in, (_dx + _G_RANGE) * (2 * _G_RANGE + 1) + (_dy + _G_RANGE), -1).astype(np.int32)
_VALID = (_COUP >= 0).astype(np.int8)

# Irreducible Brillouin-zone sampling (square lattice, units of 2*pi/a).
# Triangle: 0 <= kx <= ky, kx + ky <= 0.5.
_KPOINTS = [(0.0, 0.0), (0.25, 0.25), (0.0, 0.5)]
for i in range(4):
    for j in range(4):
        u = i / 3.0
        v = j / 3.0
        if u + v <= 1.0:
            _KPOINTS.append((u * 0.25, u * 0.25 + v * 0.5))
_KPOINTS = np.array(_KPOINTS, dtype=np.float64)
_KX = _KPOINTS[:, 0]
_KY = _KPOINTS[:, 1]


def _enforce_symmetry(cell):
    """Project a 32x32 array onto the full square-symmetry group (union rule)."""
    c = np.array(cell, dtype=np.int8)
    c = np.maximum(c, c[::-1, :])          # vertical mirror
    c = np.maximum(c, c[:, ::-1])          # horizontal mirror
    c = np.maximum(c, c.T)                 # transpose
    return c


def _make_4rod_cell(r=0.15, rr=0.0):
    """4 symmetric dielectric rods at (+/-0.25, +/-0.25); optional central rod."""
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    cx = np.array([0.25, -0.25, 0.25, -0.25])
    cy = np.array([0.25, 0.25, -0.25, -0.25])
    for i in range(_SIZE):
        x = (i + 0.5) / _SIZE - 0.5
        for j in range(_SIZE):
            y = (j + 0.5) / _SIZE - 0.5
            hit = False
            for a in range(4):
                dx = x - cx[a]
                dy = y - cy[a]
                if dx * dx + dy * dy < r * r:
                    hit = True
                    break
            if (not hit) and rr > 0.0 and x * x + y * y < rr * rr:
                hit = True
            cell[i, j] = 1 if hit else 0
    return _enforce_symmetry(cell)


def _coeffs_from_cell(cell):
    """Exact Fourier coefficients of 1/eps for a binary cell (flat 1024 array)."""
    g = 2.0 * cell - 1.0
    f = np.fft.fft2(g).ravel()
    c = np.zeros(_NPW, dtype=np.complex128)
    c[0] = _EPS_INV_MEAN
    c[1:] = _EPS_INV_AMP * f[1:]
    return c


def _score_cell(cell):
    """Gap-midgap ratio 2*(min w5 - max w4)/(min w5 + max w4) over the BZ sample."""
    coeffs = _coeffs_from_cell(cell)
    # S (1/eps) matrix, Hermitian, via the coupling table.
    S = coeffs[_COUP]
    S[_COUP < 0] = 0.0
    S = 0.5 * (S + S.conj().T)
    Sp = csr_matrix(S)

    min_w5 = float("inf")
    max_w4 = -float("inf")
    for kx, ky in zip(_KX, _KY):
        kp = _GV + np.array([kx, ky])
        d = kp[:, 0] ** 2 + kp[:, 1] ** 2
        Dp = csr_matrix(np.diag(d))
        try:
            vals = eigsh(Sp, k=5, sigma=0.0, which="LM",
                         M=Dp, maxiter=4000, tol=1e-9, v0=np.ones(_NPW))
            vals = np.sort(np.real(vals))
        except Exception:
            try:
                A = Sp @ np.linalg.inv(S + 1e-9 * np.eye(_NPW))
                A = 0.5 * (A + A.conj().T)
                vals = np.sort(np.linalg.eigvalsh(A))
            except Exception:
                continue
        if len(vals) < 5:
            continue
        w4 = math.sqrt(max(vals[3], 0.0))
        w5 = math.sqrt(max(vals[4], 0.0))
        if w4 > max_w4:
            max_w4 = w4
        if w5 < min_w5:
            min_w5 = w5

    if min_w5 == float("inf") or max_w4 <= 0.0:
        return -1.0
    denom = min_w5 + max_w4
    if denom <= 0.0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / denom


def _orbit(i, j):
    N = _SIZE
    pts = [(i, j), (j, N - 1 - i), (N - 1 - i, N - 1 - j), (N - 1 - j, i),
           (i, N - 1 - j), (N - 1 - i, j), (j, i), (N - 1 - j, N - 1 - i)]
    return list(dict.fromkeys(pts))


def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell (returns nested list)."""
    start = time.time()
    deadline = start + 222.0  # safety margin before the 230 s kill

    size = problem.get("pixels", 32)
    if size != _SIZE:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.04:
                    cell[i][j] = 1
        return cell

    rng = np.random.default_rng(12345)

    # Multi-start seeds: 4-rod geometry (best inspiration) with a few radius/centre variants.
    seeds = [
        _make_4rod_cell(0.15, 0.0),
        _make_4rod_cell(0.16, 0.0),
        _make_4rod_cell(0.14, 0.0),
        _make_4rod_cell(0.15, 0.10),
        _make_4rod_cell(0.13, 0.0),
    ]
    best_cell = None
    best_score = -np.inf
    for s in seeds:
        if time.time() > deadline - 40.0:
            break
        sc = _score_cell(s)
        if sc > best_score:
            best_score = sc
            best_cell = s.copy()

    # Pixel-level orbit-flip local search with restarts.
    no_improve = 0
    max_no_improve = 45
    while time.time() < deadline - 6.0:
        i = int(rng.integers(_SIZE))
        j = int(rng.integers(_SIZE))
        orbit = _orbit(i, j)
        for (r, c) in orbit:
            best_cell[r, c] = 1 - best_cell[r, c]
        sc = _score_cell(best_cell)
        if sc > best_score:
            best_score = sc
            no_improve = 0
        else:
            for (r, c) in orbit:
                best_cell[r, c] = 1 - best_cell[r, c]
            no_improve += 1
        if no_improve > max_no_improve:
            # restart: perturb the best cell with a few random flips
            for _ in range(6):
                ri = int(rng.integers(_SIZE))
                rj = int(rng.integers(_SIZE))
                for (r, c) in _orbit(ri, rj):
                    best_cell[r, c] = 1 - best_cell[r, c]
            no_improve = 0

    cell = _enforce_symmetry(best_cell)
    return cell.tolist()
