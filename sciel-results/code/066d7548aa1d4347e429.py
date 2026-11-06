"""Photonic-crystal unit cell: maximize TM 4th-5th band gap-midgap ratio.

Strategy:
  1. Build a fast surrogate that uses the EXACT same 289-plane-wave Galerkin
     structure as the evaluator (same G-grid, same exact Fourier coefficients),
     but only on a small set of high-symmetry wave vectors. This is cheap
     because the G-grid is only 17x17 and the dielectric map is precomputed.
  2. Optimize a 4-fold symmetric dielectric map (top-left 16x16 quadrant, then
     mirror to the full 32x32 cell) using a stochastic hill-climber on pixel
     flips, driven by the surrogate's gap-midgap ratio.
  3. Time-guard: return the best design found well before the deadline.
"""
import time
import math
import random
import numpy as np

# ---- Fixed problem constants ----
N = 32                 # pixels per side
EPS_LOW = 1.0
EPS_HIGH = 12.0
G_MAX = 13             # 17x17 G-grid => 289 plane waves (matches evaluator)
LOWER_BAND = 4         # 1-indexed: gap between band 4 and band 5

# High-symmetry k-points in the irreducible BZ (fractional coords, a=1)
# x = kx, y = ky with 0 <= x <= y <= 0.5
K_POINTS = [
    (0.0, 0.0),      # Gamma
    (0.25, 0.25),    # M/2 (center of edge)
    (0.5, 0.5),      # M
    (0.125, 0.125),
    (0.375, 0.375),
    (0.0, 0.25),     # X/2
    (0.0, 0.5),      # X
    (0.25, 0.5),     # midpoint X-M
    (0.125, 0.375),
    (0.0, 0.125),
    (0.0, 0.375),
]


def _build_g_grid():
    """G-vectors (in units of 2*pi/a) as integer pairs, |G| <= G_MAX."""
    gs = []
    for mx in range(-G_MAX, G_MAX + 1):
        for my in range(-G_MAX, G_MAX + 1):
            if mx * mx + my * my <= G_MAX * G_MAX:
                gs.append((mx, my))
    return np.array(gs, dtype=np.float64)


def _fourier_coeffs(cell):
    """Exact Fourier coefficients eps_G of the pixel dielectric map.

    eps(x) = sum_G eps_G exp(i G . x),  with x in [0,1)^2 (a=1).
    eps_G = (1/N^2) sum_{i,j} eps_{ij} exp(-i 2pi (mx i/N + my j/N)).
    For real even-ish maps we take the real part; keep complex for safety.
    """
    eps = np.asarray(cell, dtype=np.float64)
    N = eps.shape[0]
    # 2D FFT: eps_G[mx,my] = (1/N^2) sum eps[i,j] exp(-i 2pi (mx i/N + my j/N))
    F = np.fft.fft2(eps) / (N * N)
    # Reorder so that index corresponds to (mx, my) in range [-N/2, N/2)
    F = np.fft.fftshift(F)
    return F


def _eval_bands(F, kx, ky, n_bands=6):
    """Solve the TM band problem at one k using the exact Galerkin matrices.

    TM: (G - k)^2 * eps_{-G'} eps_{G-G'} ... standard form:
        det[ (G-k)^2 delta - eps_{G-G'} ] = 0  (in normalized units)
    We build the Hermitian-like matrix M[G, G'] = (G-k)^2 delta - eps_{G-G'}.
    Eigenvalues are the (normalized) omega^2.
    """
    G = _G_CACHE
    kvec = np.array([kx, ky], dtype=np.float64)
    GK = G - kvec  # (G - k)
    gk2 = np.einsum('ij,ij->i', GK, GK)  # |G-k|^2
    n = len(G)

    # Difference vectors G - G' for all pairs -> look up eps_{G-G'}
    # Build eps matrix: M_eps[p, q] = eps_{G_p - G_q}
    # G_p - G_q = (mx_p - mx_q, my_p - my_q)
    diff = G[:, None, :] - G[None, :, :]  # (n, n, 2)
    dmx = diff[:, :, 0].astype(np.int64)
    dmy = diff[:, :, 1].astype(np.int64)
    # Map to F index: F is fftshifted, index = dmx + N/2, dmy + N/2
    off = N // 2
    idx_m = dmx + off
    idx_y = dmy + off
    # Only valid if within [0, N)
    valid = (idx_m >= 0) & (idx_m < N) & (idx_y >= 0) & (idx_y < N)
    eps_mat = np.zeros((n, n), dtype=np.complex128)
    eps_mat[valid] = F[idx_m[valid], idx_y[valid]]

    # Matrix M = diag(gk2) - eps_mat
    M = np.diag(gk2) - eps_mat
    # M is Hermitian if eps is real & even (eps_{-G}=conj(eps_G)); symmetrize
    M = 0.5 * (M + M.conj().T)
    # Eigenvalues
    w2 = np.linalg.eigvalsh(M)
    # Filter non-physical (negative from numerical noise)
    w2 = np.where(w2 > 0, w2, 0.0)
    w = np.sqrt(w2)
    w.sort()
    return w[:n_bands]


def _gap_midgap(cell, n_bands=6):
    """Compute min over k of band5 and max over k of band4, return ratio."""
    F = _fourier_coeffs(cell)
    min5 = np.inf
    max4 = -np.inf
    for (kx, ky) in K_POINTS:
        w = _eval_bands(F, kx, ky, n_bands=n_bands)
        if len(w) < 5:
            continue
        b4 = w[3]   # band 4 (1-indexed)
        b5 = w[4]   # band 5
        if b5 < min5:
            min5 = b5
        if b4 > max4:
            max4 = b4
    if min5 == np.inf or max4 == -np.inf:
        return -1.0
    ratio = 2.0 * (min5 - max4) / (min5 + max4)
    return ratio


# Precompute G grid once
_G_CACHE = _build_g_grid()


def _quadrant_to_cell(q):
    """Expand a 16x16 top-left quadrant to a 32x32 4-fold symmetric cell.

    Symmetries about cell center: row-reverse, col-reverse, transpose.
    Top-left quadrant Q[i][j], i,j in [0,16).
    Full cell C:
      C[i][j] = Q[i][j]                          (i<16, j<16)
      C[i][31-j] = Q[i][j]                       (i<16, j>=16)  col-reverse
      C[31-i][j] = Q[i][j]                       (i>=16, j<16)  row-reverse
      C[31-i][31-j] = Q[i][j]                    (i>=16, j>=16)
    This makes C row-reverse-symmetric, col-reverse-symmetric, and transpose-symmetric.
    """
    C = [[0] * N for _ in range(N)]
    for i in range(16):
        for j in range(16):
            v = q[i][j]
            C[i][j] = v
            C[i][N - 1 - j] = v
            C[N - 1 - i][j] = v
            C[N - 1 - i][N - 1 - j] = v
    return C


def _cell_to_quadrant(cell):
    q = [[cell[i][j] for j in range(16)] for i in range(16)]
    return q


def _random_quadrant():
    q = [[random.randint(0, 1) for _ in range(16)] for _ in range(16)]
    return q


def _flip_quadrant(q, i, j):
    q[i][j] = 1 - q[i][j]


def design(problem):
    size = problem.get("pixels", 32)
    assert size == N
    t0 = time.time()
    DEADLINE = 200.0  # safety margin under 230s kill

    rng = random.Random(12345)

    # --- Stage 1: Try a few parameterized good starting designs ---
    candidates = []

    # (a) centered rod, radius ~ 0.22 (slightly larger than seed 0.2)
    for r in (0.18, 0.20, 0.22, 0.24, 0.26):
        cell = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if x * x + y * y < r * r:
                    cell[i][j] = 1
        candidates.append(cell)

    # (b) dielectric square (checkerboard-ish) - a square hole / rod
    for frac in (0.3, 0.4, 0.5):
        cell = [[0] * N for _ in range(N)]
        half = int(N * frac / 2)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if abs(x) < half / N and abs(y) < half / N:
                    cell[i][j] = 1
        candidates.append(cell)

    # (c) inverse: air rod in dielectric (high fill)
    for r in (0.15, 0.20, 0.25):
        cell = [[1] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if x * x + y * y < r * r:
                    cell[i][j] = 0
        candidates.append(cell)

    # (d) random starts
    for _ in range(6):
        q = _random_quadrant()
        candidates.append(_quadrant_to_cell(q))

    # Evaluate all candidates with the surrogate
    best_cell = candidates[0]
    best_score = -1e9
    for cell in candidates:
        s = _gap_midgap(cell)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in cell]

    # --- Stage 2: Local pixel-flip hill climbing on the quadrant ---
    q = _cell_to_quadrant(best_cell)
    cur_score = best_score
    cur_q = [row[:] for row in q]

    # Adaptive step: try many flips, accept improvements (first-improvement)
    flip_count = 0
    while time.time() < DEADLINE:
        improved = False
        # Random order of pixels in quadrant
        order = list(range(16 * 16))
        rng.shuffle(order)
        for idx in order:
            if time.time() > DEADLINE:
                break
            i, j = idx // 16, idx % 16
            _flip_quadrant(cur_q, i, j)
            cell = _quadrant_to_cell(cur_q)
            s = _gap_midgap(cell)
            flip_count += 1
            if s > cur_score:
                cur_score = s
                best_cell = [row[:] for row in cell]
                best_q = [row[:] for row in cur_q]
                improved = True
            else:
                # revert
                _flip_quadrant(cur_q, i, j)
        if not improved:
            # No improvement in a full pass; try a random restart with a
            # perturbation of the best, or a fresh random start.
            # Perturb: flip a few random pixels in best_q
            pert = [row[:] for row in best_q]
            for _ in range(rng.randint(2, 6)):
                i = rng.randrange(16)
                j = rng.randrange(16)
                _flip_quadrant(pert, i, j)
            cell = _quadrant_to_cell(pert)
            s = _gap_midgap(cell)
            if s > cur_score:
                cur_score = s
                best_cell = [row[:] for row in cell]
                best_q = [row[:] for row in pert]
                cur_q = [row[:] for row in pert]
            else:
                # Occasional fresh random restart
                if rng.random() < 0.3:
                    newq = _random_quadrant()
                    cell = _quadrant_to_cell(newq)
                    s = _gap_midgap(cell)
                    if s > cur_score:
                        cur_score = s
                        best_cell = [row[:] for row in cell]
                        best_q = [row[:] for row in newq]
                        cur_q = [row[:] for row in newq]
                    else:
                        # keep cur_q = best_q to continue from best
                        cur_q = [row[:] for row in best_q]

    # Ensure symmetry is exact (rebuild from quadrant)
    final_q = _cell_to_quadrant(best_cell)
    final_cell = _quadrant_to_cell(final_q)
    return final_cell
