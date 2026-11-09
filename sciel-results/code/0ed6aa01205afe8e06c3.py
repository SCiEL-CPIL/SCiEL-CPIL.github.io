"""Direct pixel-level search over the full 32x32 unit cell using a fast
plane-wave surrogate band-gap estimator. Structural change from the seed:
no analytical parameterization -- we optimise the actual pixel map."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_EPS_HIGH = 12.0
_EPS_LOW = 1.0
_EPS0 = 1.0
_EPS0_INV = 1.0 / _EPS0
_NPW = 9            # fast surrogate plane waves (9x9 = 81)
_KPTS = 7           # 7 k-points along M-X-G
_BUDGET = 200.0     # safety margin well under the 230 s kill limit
_START = time.time()


def _remaining():
    return _BUDGET - (time.time() - _START)


def _make_eps_grid(cell, n):
    """Map the 32x32 pixel map onto an n x n grid via nearest-neighbour."""
    eps = np.empty((n, n), dtype=float)
    for i in range(n):
        row = int(round((i + 0.5) * 32.0 / n))
        if row >= 32:
            row = 31
        for j in range(n):
            col = int(round((j + 0.5) * 32.0 / n))
            if col >= 32:
                col = 31
            eps[i, j] = _EPS_HIGH if cell[row, col] else _EPS_LOW
    return eps


def _fourier_coeffs(eps, npw):
    """Exact Fourier coefficients of the periodic pixel dielectric map."""
    N = eps.shape[0]
    mx = (npw - 1) // 2
    G = np.arange(-mx, mx + 1, dtype=float)
    c = np.zeros((npw, npw), dtype=complex)
    rows = np.arange(N)
    for mi in range(npw):
        gx = G[mi]
        phi = np.exp(2j * math.pi * gx * rows / N)
        for mj in range(npw):
            gy = G[mj]
            if gx == 0.0 and gy == 0.0:
                c[mi, mj] = eps.mean()
                continue
            psi = np.exp(2j * math.pi * gy * rows / N)
            c[mi, mj] = (eps * np.outer(phi, psi)).mean()
    return c, G


def _tm_bands(c, G, npw, kx, ky):
    """Solve the TM eigenvalue problem; return sorted eigenvalues."""
    mx = (npw - 1) // 2
    idx = np.arange(npw)
    gx = G[idx]
    gy = G[idx]
    # K^2 for each plane wave (vectorised over the 2D grid)
    Kx = kx + gx[:, None] + np.zeros((1, npw))
    Ky = ky + np.zeros((npw, 1)) + gy[None, :]
    K2 = (Kx ** 2 + Ky ** 2).reshape(-1) / _EPS0
    A = np.diag(K2).astype(complex)
    # off-diagonal coupling: A[ic, ia] += c[da+mx, db+mx] * K2[ic]
    for a in range(npw):
        for b in range(npw):
            da = a - mx
            db = b - mx
            if da == 0 and db == 0:
                continue
            coeff = c[da + mx, db + mx]
            if coeff == 0.0:
                continue
            ia = a * npw + b
            for ic in range(npw * npw):
                A[ic, ia] += coeff * K2[ic]
    w, _ = np.linalg.eigh(A)
    return np.sort(np.real(w))


def _bandgap_ratio(cell):
    """Compute 2*(min w5 - max w4)/(min w5 + max w4) over the k-path."""
    eps = _make_eps_grid(cell, _NPW)
    c, G = _fourier_coeffs(eps, _NPW)
    kpts = []
    for t in np.linspace(0.0, 1.0, _KPTS):
        if t < 0.5:
            s = t / 0.5
            kx = 0.5
            ky = 0.5 * (1 - s)
        else:
            s = (t - 0.5) / 0.5
            kx = 0.5 * (1 - s)
            ky = 0.0
        kpts.append((kx, ky))
    max_w4 = 0.0
    min_w5 = np.inf
    for kx, ky in kpts:
        w = _tm_bands(c, G, _NPW, kx, ky)
        if len(w) < 5:
            continue
        max_w4 = max(max_w4, w[3])
        min_w5 = min(min_w5, w[4])
    if not np.isfinite(min_w5) or min_w5 + max_w4 <= 0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _seed_cell():
    """Centred dielectric rod of radius 0.25."""
    cell = np.zeros((32, 32), dtype=int)
    for i in range(32):
        for j in range(32):
            x = (i + 0.5) / 32.0 - 0.5
            y = (j + 0.5) / 32.0 - 0.5
            if x * x + y * y < 0.25 ** 2:
                cell[i, j] = 1
    return cell


def _local_search(cell, best_score):
    """Coarse hill-climbing: flip pixels in the TL quadrant, keep if better."""
    h = 32 // 2
    positions = [(i, j) for i in range(h) for j in range(h)]
    rng = np.random.default_rng(12345)
    order = list(positions)
    rng.shuffle(order)
    for i, j in order:
        if _remaining() < 12.0:
            break
        cell[i, j] = 1 - cell[i, j]
        s = _bandgap_ratio(cell)
        if s > best_score:
            best_score = s
        else:
            cell[i, j] = 1 - cell[i, j]  # revert
    return cell, best_score


def design(problem):
    size = problem["pixels"]
    cell = _seed_cell()
    best_score = _bandgap_ratio(cell)
    best_cell = cell.copy()

    # A few full passes of hill-climbing within budget.
    for _ in range(12):
        if _remaining() < 25.0:
            break
        cell, best_score = _local_search(cell, best_score)
        if best_score >= _bandgap_ratio(best_cell) + 1e-12:
            best_cell = cell.copy()

    out = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return out
# EVOLVE-BLOCK-END
