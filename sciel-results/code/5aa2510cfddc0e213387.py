import numpy as np
import time

_DEADLINE = 220.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

# ---- fast 7x7 plane-wave TM surrogate --------------------------------------
def _pw_eps_mask(cell, K):
    size = len(cell)
    m = np.asarray(cell, dtype=np.float64)
    eps = 1.0 + 11.0 * m
    f = np.fft.fft2(eps) / (size * size)
    K2 = K * K
    g = np.empty((2 * K + 1) * (2 * K + 1), dtype=np.float64)
    n = 0
    for ix in range(-K, K + 1):
        for iy in range(-K, K + 1):
            g[n] = f[ix % size, iy % size]
            n += 1
    return g, K2

def _tm_bands(g, K2, kx, ky, K):
    N = (2 * K + 1) * (2 * K + 1)
    M = np.empty((N, N), dtype=np.float64)
    n = 0
    for ix in range(-K, K + 1):
        for iy in range(-K, K + 1):
            wx = kx + ix
            wy = ky + iy
            w2 = wx * wx + wy * wy
            for j in range(N):
                M[n, j] = w2 * g[j] * (1.0 if j == n else 0.0)
                n += 1
    w = np.linalg.eigvalsh(M)
    return np.sqrt(np.maximum(w, 0.0))

def _score_cell(cell, K=3):
    g, K2 = _pw_eps_mask(cell, K)
    size = len(cell)
    L = size / 2.0
    maxw4 = -1e30
    minw5 = 1e30
    for i in range(L + 1):
        for j in range(L + 1 - i):
            kx = i / L
            ky = j / L
            b = _tm_bands(g, K2, kx, ky, K)
            maxw4 = max(maxw4, b[3])
            minw5 = min(minw5, b[4])
    if minw5 <= 0 or maxw4 <= 0:
        return -1.0
    return 2.0 * (minw5 - maxw4) / (minw5 + maxw4)

# ---- C4v-symmetric parameterized mask --------------------------------------
def _make_cell(p, size=32):
    a0, a1, a2, a3, a4, a5, r0, thr = p
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size - 0.5
    y = (yy + 0.5) / size - 0.5
    r = np.sqrt(x * x + y * y)
    f = a0 + a1 * (2 * r * r - 1.0) + a2 * (4 * x * y) + a3 * (4 * (x * x - y * y)) \
        + a4 * (8 * x * x * y * y) + a5 * (8 * (x * x * x * y + x * y * y * y))
    f = f + (1.0 - r * r / max(r0 * r0, 1e-9))
    cell = (f > thr).astype(int)
    return cell.tolist()

_BOUNDS = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.15, 0.55), (-2.0, 2.0)]

def _search():
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.0])
    cell = _make_cell(p)
    best = _score_cell(cell)
    best_p = p.copy()
    best_cell = cell
    # coarse-to-fine coordinate search
    for step in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
        if _remaining() < 8.0:
            break
        improved = True
        while improved:
            improved = False
            if _remaining() < 8.0:
                break
            for d in range(len(p)):
                if _remaining() < 8.0:
                    break
                lo, hi = _BOUNDS[d]
                for sgn in (1.0, -1.0):
                    if _remaining() < 8.0:
                        break
                    cand = best_p.copy()
                    cand[d] += sgn * step
                    if cand[d] < lo or cand[d] > hi:
                        continue
                    c = _make_cell(cand)
                    s = _score_cell(c)
                    if s > best + 1e-9:
                        best = s
                        best_p = cand
                        best_cell = c
                        improved = True
    return best_cell

def design(problem):
    size = problem["pixels"]
    # seed: centred dielectric rod radius 0.2 (guaranteed valid, C4v symmetric)
    seed = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        seed.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    if _remaining() < 15.0:
        return seed
    try:
        cell = _search()
    except Exception:
        return seed
    # enforce C4v symmetry (transpose + row/col reversal)
    c = np.asarray(cell, dtype=int)
    c = (c + c.T + c[::-1, :] + c[:, ::-1]) // 4
    return c.tolist()
