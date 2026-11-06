"""Photonic-crystal 4-5 TM band-gap designer.

Structural change vs. the seed: the dielectric map is parameterized as a
superposition of low-order Fourier harmonics that directly couple the 4th and
5th TM bands, and the amplitudes/phases of those modes (plus a DC level) are
optimized to maximize the 4-5 gap-midgap ratio.  A small, exact Fourier-
coefficient plane-wave TM band-structure solver is used as the objective.
"""
import time
import math
import numpy as np

_DEADLINE = 220.0  # wall-clock safety margin (evaluator kills at 230 s)


# ---------------------------------------------------------------------------
# Symmetry + binarization
# ---------------------------------------------------------------------------
def _symmetrize(m):
    """Enforce square symmetry about the cell centre (transpose + 180 deg)."""
    m = np.asarray(m, dtype=np.float64)
    m = (m + m.T + m[::-1, ::-1] + m.T[::-1, ::-1]) / 4.0
    return m


def _make_cell(params, size, eps_low, eps_high):
    """Build a symmetric 0/1 cell from harmonic parameters.

    params = [dc, a20, p20, a02, p02, a22, p22, a40, p40, a04, p04, a44, p44]
    """
    dc = params[0]
    i2, j2 = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    i = (i2 + 0.5) / size - 0.5
    j = (j2 + 0.5) / size - 0.5
    f = np.full((size, size), dc, dtype=np.float64)
    f += params[1] * np.cos(2 * np.pi * i + params[2])
    f += params[3] * np.cos(2 * np.pi * j + params[4])
    f += params[5] * np.cos(2 * np.pi * (i + j) + params[6])
    f += params[7] * np.cos(4 * np.pi * i + params[8])
    f += params[9] * np.cos(4 * np.pi * j + params[10])
    f += params[11] * np.cos(4 * np.pi * (i + j) + params[12])
    f = _symmetrize(f)
    thr = 0.5 * (eps_low + eps_high)
    eps = np.where(f > 0, eps_high, eps_low)
    return ((eps - eps_low) / (eps_high - eps_low)).astype(int)


# ---------------------------------------------------------------------------
# Exact Fourier coefficients of 1/eps (pixel map)
# ---------------------------------------------------------------------------
def _fourier_coeffs(eps, N, size):
    """Return dict {(mx,my): c} for |mx|,|my| <= N of the Fourier series of 1/eps.

    c_{mn} = sum_{i,j} (1/eps_{ij}) exp(-i 2 pi (m i + n j)/size) / size^2
    """
    inv = 1.0 / eps.astype(np.float64)
    i2, j2 = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    c = {}
    for mx in range(-N, N + 1):
        for my in range(-N, N + 1):
            c[(mx, my)] = np.sum(inv * np.exp(-2j * np.pi * (mx * i2 + my * j2) / size)) / (size * size)
    return c


# ---------------------------------------------------------------------------
# TM plane-wave Galerkin band structure
# ---------------------------------------------------------------------------
def _band_structure_tm(eps, kx, ky, N, n_bands, size):
    """Return the lowest n_bands TM eigenfrequencies (in units of a/2pi) at (kx,ky).

    kx,ky are in units of 2pi/a.  Uses the 1/eps operator (TM polarization).
    """
    c = _fourier_coeffs(eps, N, size)
    m = list(range(-N, N + 1))
    G = np.array([(mx, my) for mx in m for my in m], dtype=np.float64)
    nG = len(G)
    kvec = np.array([kx, ky], dtype=np.float64)
    K = G + kvec
    K2 = np.sum(K * K, axis=1)
    # Operator: (K . K') * c[K - K']
    D = np.empty((nG, nG), dtype=np.float64)
    for gi in range(nG):
        for gj in range(nG):
            d = (G[gi] - G[gj]).astype(int)
            D[gi, gj] = np.dot(K[gi], K[gj]) * c.get((d[0], d[1]), 0.0).real
    vals = np.linalg.eigvalsh(0.5 * D)
    vals = np.sort(vals)
    return np.sqrt(np.maximum(vals[:n_bands], 0.0))


# ---------------------------------------------------------------------------
# Objective: 4-5 gap-midgap ratio
# ---------------------------------------------------------------------------
def _gap_ratio(eps, size):
    N = 13
    # High-symmetry + interior points of the irreducible BZ (in units of 2pi/a).
    # Irreducible wedge: 0 <= ky <= kx, kx in [0, 0.5], plus the X->M edge.
    kpts = []
    for t in np.linspace(0.0, 0.5, 9):            # Gamma -> X  (ky=0)
        kpts.append((t, 0.0))
    for t in np.linspace(0.0, 0.5, 9):            # X -> M     (kx=0.5)
        kpts.append((0.5, t))
    for t in np.linspace(0.0, 0.5, 7):            # M -> Gamma (kx=ky)
        kpts.append((t, t))
    for t in np.linspace(0.1, 0.4, 4):            # interior of wedge
        for s in np.linspace(0.05, 0.35, 4):
            kpts.append((t, s))
    max4 = -np.inf
    min5 = np.inf
    for (kx, ky) in kpts:
        w = _band_structure_tm(eps, kx, ky, N, 5, size)
        if w.size < 5:
            continue
        max4 = max(max4, w[3])
        min5 = min(min5, w[4])
    if not np.isfinite(min5) or min5 + max4 <= 0:
        return -1.0
    return 2.0 * (min5 - max4) / (min5 + max4)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def _initial_params(rng):
    p = [0.5]
    # (2,0),(0,2),(2,2) primary; (4,0),(0,4),(4,4) secondary
    amps = [0.35, 0.35, 0.30, 0.12, 0.12, 0.10]
    for a in amps:
        p.append(a * (0.5 + 0.5 * rng.random()))
        p.append(2 * np.pi * rng.random())
    return p


def _optimize(params0, rng, max_iter, deadline):
    from scipy.optimize import minimize
    x = np.array(params0, dtype=np.float64)
    best = (None, -np.inf)

    def obj(xv):
        size = 32
        cell = _make_cell(xv, size, 1.0, 12.0)
        r = _gap_ratio(cell, size)
        return -r

    res = minimize(obj, x, method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 0.01, "fatol": 1e-4,
                            "adaptive": True})
    if res.x is not None and np.isfinite(res.fun):
        r = -res.fun
        if r > best[1]:
            best = (res.x, r)
    return best


def design(problem):
    global _DEADLINE
    t0 = time.time()
    size = int(problem["pixels"])
    eps_low = float(problem.get("eps_low", 1.0))
    eps_high = float(problem.get("eps_high", 12.0))
    rng = np.random.default_rng(20240517)

    best_cell = None
    best_score = -np.inf

    # Seed with the strong measured four-rod configuration as a starting point.
    def four_rod_cell():
        i2, j2 = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        i = (i2 + 0.5) / size - 0.5
        j = (j2 + 0.5) / size - 0.5
        val = 0
        for sx in (0.25, -0.25):
            for sy in (0.25, -0.25):
                val = np.maximum(val, 1.0 - ((i - sx) ** 2 + (j - sy) ** 2) / (0.13 ** 2))
        return (val > 0.5).astype(int)

    fr = four_rod_cell()
    s0 = _gap_ratio(fr, size)
    best_cell, best_score = fr, s0

    # Random restarts + Nelder-Mead over the harmonic parameter space.
    n_restarts = 24
    max_iter = 120
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 40.0:
            break
        p0 = _initial_params(rng)
        # bias the DC toward the four-rod dielectric filling (~0.35)
        p0[0] = 0.30 + 0.15 * rng.random()
        x, sc = _optimize(p0, rng, max_iter, _DEADLINE)
        if x is not None and sc > best_score:
            best_score = sc
            best_cell = _make_cell(x, size, eps_low, eps_high)
        if time.time() > _DEADLINE - 30.0:
            break

    # Guarantee the return type is a plain nested list of 0/1 ints.
    arr = np.asarray(best_cell, dtype=int)
    return [[int(arr[i][j]) for j in range(size)] for i in range(size)]
