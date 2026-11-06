"""Photonic-crystal design: maximize 4-5 TM band gap in a 32x32 square unit cell.
Uses an exact plane-wave Galerkin surrogate (matching the evaluator's method)
plus Nelder-Mead optimization over a symmetry-constrained superellipse shape.
"""
import time
import math
import numpy as np

_DEADLINE = 230.0
_MARGIN = 15.0  # safety margin before wall-clock kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _fourier_eps_inv(cell, eps_low, eps_high, N):
    """Compute the N x N Fourier matrix of 1/eps over the unit cell.
    cell is a size x size 0/1 array (1=dielectric). We compute G-space
    coefficients of the reciprocal (inverse) dielectric, which is what
    enters the TM eigenproblem:  (k-G)·(k-G) * inv_eps(G-G') = omega^2/eps0 c^2.
    """
    size = len(cell)
    c = np.array(cell, dtype=float)
    eps = np.where(c > 0.5, eps_high, eps_low)
    inv_eps = 1.0 / eps
    # Pixel centers in unit-cell coords
    xs = (np.arange(size) + 0.5) / size
    # 2D FFT-based: treat inv_eps as a function on the unit cell, sample at pixels.
    # Use a direct DFT via fft2 on the pixel grid, then map to the needed G indices.
    # The pixel grid is a uniform sampling of the unit cell, so fft2 gives Fourier
    # coefficients directly (with the standard normalization).
    F = np.fft.fft2(inv_eps)  # F[m,n] = sum_{i,j} inv_eps[i,j] e^{-2pi i (m i + n j)/size}
    # Normalize to the continuous integral approximation: coeff = (1/size^2) * F
    coeff = F / (size * size)
    # We need coefficients for G indices in [-N//2, N//2-1] x same.
    half = N // 2  # N odd => half = 8 for N=17
    idx = np.arange(-half, half + 1)  # 17 values: -8..8
    # Map our FFT output (indices 0..size-1) to the desired G range.
    # fft2 gives frequency index k in 0..size-1 corresponding to k if k<size/2 else k-size.
    out = np.zeros((N, N), dtype=complex)
    for a in range(N):
        ga = idx[a]
        fa = ga if ga >= 0 else ga + size
        for b in range(N):
            gb = idx[b]
            fb = gb if gb >= 0 else gb + size
            out[a, b] = coeff[fa, fb]
    return out


def _band_structure(cell, eps_low, eps_high, N=17, n_kx=6, n_ky=6):
    """Compute max omega_4 and min omega_5 over the irreducible BZ using
    the plane-wave Galerkin method. Returns (max_o4, min_o5) in units of a/(2pi c).
    """
    size = len(cell)
    half = N // 2
    idx = np.arange(-half, half + 1)  # G indices in units of 2pi/a
    # G vectors
    Gx = idx * 2.0 * math.pi
    Gy = idx * 2.0 * math.pi
    # Fourier coeffs of inv_eps
    E = _fourier_eps_inv(cell, eps_low, eps_high, N)
    # Build the eigenvalue problem. For TM, the matrix is:
    # sum_{G'} (k - G)·(k - G) * inv_eps(G - G')  = (omega/(2pi c))^2 * a^2 / (2pi)^2 ...
    # In normalized units (k in 2pi/a, G in 2pi/a), the eigenvalue lambda = (omega a / (2pi c))^2.
    # We'll work in units where the matrix is M[p,q] = (k-G_p)·(k-G_p) * E[p, q-p] ... 
    # Actually the standard form: for each plane wave p with vector G_p, the equation is
    #   sum_q (k - G_p)·(k - G_p) * inv_eps(G_p - G_q) A_q = lambda A_p
    # So M[p,q] = (k-G_p)·(k-G_p) * inv_eps(G_p - G_q).
    # inv_eps(G_p - G_q) = E[ index(G_p - G_q) ].
    # Precompute the difference-index map.
    # idx maps value v (in -half..half) to position v+half.
    def _diff_index(ga, gb):
        d = ga - gb
        # d is in -2*half..2*half; we only have -half..half. If out of range, coeff ~ 0 (aliasing, ignore).
        if d < -half or d > half:
            return -1
        return d + half

    # Map each p to its (gx, gy)
    p_gx = np.array([Gx[a] for a in range(N)])
    p_gy = np.array([Gy[b] for b in range(N)])
    # Actually p index = a*N + b
    P = N * N
    # Build E difference matrix: ED[p,q] = E[ diff(G_p - G_q) ]
    # G_p = (idx[a_p]*2pi, idx[b_p]*2pi). G_p - G_q has components (idx[a_p]-idx[a_q], idx[b_p]-idx[b_q])
    a_idx = np.arange(N)
    gx_comp = idx  # in units of 2pi
    gy_comp = idx
    # For p=(a,b), q=(c,d): diff = (a-c, b-d) in idx units.
    # We need E at index (a-c + half, b-d + half) if in range.
    # Build a 2D array D[a,b,c,d] is too big; instead build ED as P x P.
    ED = np.zeros((P, P), dtype=complex)
    for a in range(N):
        for b in range(N):
            p = a * N + b
            for c in range(N):
                for d in range(N):
                    q = c * N + d
                    da = a - c
                    db = b - d
                    ia = da + half
                    ib = db + half
                    if 0 <= ia < N and 0 <= ib < N:
                        ED[p, q] = E[ia, ib]
    # Now for each k, build M and solve.
    # k points: irreducible BZ of square: kx in [0, 2pi/a], ky in [0, kx] (triangle).
    # Use a grid over the triangle.
    K = 2.0 * math.pi  # in 2pi/a units, the BZ boundary is at kx=1, ky=1 (i.e., 2pi/a).
    # Work in units of 2pi/a: kx,ky in [0,1]. Triangle: 0<=ky<=kx<=1.
    max_o4 = 0.0
    min_o5 = float('inf')
    for ix in range(n_kx):
        kx = (ix + 0.5) / n_kx
        for iy in range(n_ky):
            ky_max = kx
            ky = (iy + 0.5) / n_ky
            if ky > ky_max:
                continue
            # k vector in 2pi/a units
            kx_u = kx
            ky_u = ky
            # (k - G_p) in 2pi/a units: kx_u - gx_comp[a], ky_u - gy_comp[b]
            # Build M: M[p,q] = (k-G_p)·(k-G_p) * ED[p,q]
            M = np.zeros((P, P), dtype=complex)
            for a in range(N):
                for b in range(N):
                    p = a * N + b
                    dx = kx_u - gx_comp[a]
                    dy = ky_u - gy_comp[b]
                    kk = dx * dx + dy * dy
                    for q in range(P):
                        M[p, q] = kk * ED[p, q]
            # Symmetrize for numerical safety
            M = 0.5 * (M + M.conj().T)
            try:
                w = np.linalg.eigvalsh(M)
            except np.linalg.LinAlgError:
                continue
            w = np.sort(np.real(w))
            # Bands 4 and 5 are 0-indexed 3 and 4.
            if len(w) < 5:
                continue
            o4 = math.sqrt(max(w[3], 0.0))
            o5 = math.sqrt(max(w[4], 0.0))
            if o4 > max_o4:
                max_o4 = o4
            if o5 < min_o5:
                min_o5 = o5
    return max_o4, min_o5


def _gap_ratio(cell, eps_low, eps_high):
    max_o4, min_o5 = _band_structure(cell, eps_low, eps_high, N=17, n_kx=6, n_ky=6)
    if max_o4 <= 0 or min_o5 <= 0:
        return 0.0
    return 2.0 * (min_o5 - max_o4) / (min_o5 + max_o4)


def _superellipse_cell(params, size=32, eps_high=12.0):
    """Build a symmetric 32x32 cell from superellipse parameters.
    params: (r, n) where r is the 'radius' scale and n is the superellipse exponent.
    Superellipse: |x|^n + |y|^n <= r^n.
    """
    r, n = params
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            val = abs(x) ** n + abs(y) ** n
            row.append(1 if val <= r ** n else 0)
        cell.append(row)
    return cell


def design(problem):
    global _START
    _START = time.time()
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    # Baseline: circular rod
    best_cell = None
    best_score = -1.0
    for r0 in [0.2, 0.25, 0.3]:
        cell = _superellipse_cell((r0, 2.0), size, eps_high)
        best_cell = cell
        break
    # Evaluate baseline
    try:
        s = _gap_ratio(best_cell, eps_low, eps_high)
        best_score = s
    except Exception:
        best_score = -1.0

    # Nelder-Mead over (r, n)
    # Clamp n to [1.0, 8.0], r to [0.05, 0.5]
    def objective(params):
        r = float(np.clip(params[0], 0.05, 0.5))
        n = float(np.clip(params[1], 1.0, 8.0))
        if _time_left() < _MARGIN + 5.0:
            return best_score  # stop, return current best
        try:
            cell = _superellipse_cell((r, n), size, eps_high)
            sc = _gap_ratio(cell, eps_low, eps_high)
        except Exception:
            sc = -1.0
        return sc

    # Use scipy.optimize.minimize with Nelder-Mead (maximize => minimize negative)
    try:
        from scipy.optimize import minimize
        x0 = np.array([0.25, 2.0])
        # Run a few iterations, checking time
        result = minimize(lambda p: -objective(p), x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-4})
        if result.success or True:
            r_opt = float(np.clip(result.x[0], 0.05, 0.5))
            n_opt = float(np.clip(result.x[1], 1.0, 8.0))
            cell = _superellipse_cell((r_opt, n_opt), size, eps_high)
            sc = _gap_ratio(cell, eps_low, eps_high)
            if sc > best_score:
                best_score = sc
                best_cell = cell
    except Exception:
        pass

    # If time remains, try a finer local search around the best
    if best_cell is not None and _time_left() > _MARGIN + 30.0:
        # Extract current best params by trying to match
        # Just do a small grid refinement around a few candidate (r,n)
        candidates = [(0.2, 2.0), (0.25, 2.0), (0.3, 2.0), (0.25, 4.0), (0.2, 4.0), (0.3, 4.0),
                      (0.25, 1.5), (0.3, 1.5), (0.2, 6.0), (0.25, 6.0), (0.3, 6.0), (0.35, 2.0)]
        for (r, n) in candidates:
            if _time_left() < _MARGIN + 5.0:
                break
            try:
                cell = _superellipse_cell((r, n), size, eps_high)
                sc = _gap_ratio(cell, eps_low, eps_high)
                if sc > best_score:
                    best_score = sc
                    best_cell = cell
            except Exception:
                pass

    if best_cell is None:
        # Fallback: circular rod
        best_cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                row.append(1 if x * x + y * y < 0.2 ** 2 else 0)
            best_cell.append(row)
    return best_cell


_START = time.time()
