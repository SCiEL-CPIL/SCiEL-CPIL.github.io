"""Design a symmetric 32x32 photonic-crystal unit cell maximizing the 4-5 TM band gap."""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 205.0   # safety margin before the 230 s wall-clock kill

# --- Plane-wave surrogate solver (TM mode) ------------------------------------
def _build_eps_grid(cell):
    """Return (eps_inv, N) for a cell; eps_inv = 1/eps on the coarse grid."""
    N = len(cell)
    arr = np.asarray(cell, dtype=np.float64)
    eps = np.where(arr > 0.5, 12.0, 1.0)
    return 1.0 / eps, N

def _fourier_coeffs(eps_inv, N, nmax):
    """Exact Fourier coefficients of eps_inv on an N x N pixel grid.
    Returns dict {(mx,my): value} for |mx|,|my| <= nmax."""
    x = np.arange(N)
    # 2D DFT via separable 1D transforms
    F = np.fft.fft2(eps_inv) / (N * N)
    # map pixel indices to integer Fourier indices centered at 0
    idx = np.fft.fftfreq(N, d=1.0 / N)  # gives integer frequencies 0..N/2, -N/2+1..-1
    coeffs = {}
    for mx in range(-nmax, nmax + 1):
        for my in range(-nmax, nmax + 1):
            coeffs[(mx, my)] = F[int(mx) % N, int(my) % N]
    return coeffs

def _pw_solve(coeffs, kx, ky, nmax):
    """Solve TM eigenvalue problem on a square lattice (a=1).
    Eigenvalue = (kx+mx)^2 + (ky+my)^2  ;  matrix couples via eps_inv Fourier coeffs.
    Returns sorted eigenvalues (omega^2 in units of c/a)."""
    # Build list of plane-wave indices
    idxs = []
    for mx in range(-nmax, nmax + 1):
        for my in range(-nmax, nmax + 1):
            idxs.append((mx, my))
    n = len(idxs)
    i2idx = {v: i for i, v in enumerate(idxs)}
    K = np.zeros((n, n))
    for i, (mx, my) in enumerate(idxs):
        for j, (px, py) in enumerate(idxs):
            c = coeffs.get((mx - px, my - py), 0.0)
            if c != 0:
                K[i, j] = c
    # RHS diagonal: (kx+mx)^2 + (ky+my)^2
    rhs = np.array([(kx + mx) ** 2 + (ky + my) ** 2 for (mx, my) in idxs])
    # Generalized eigenvalue: K @ v = lambda * rhs * v
    # Convert to standard: (rhs^-1/2 @ K @ rhs^-1/2) @ w = lambda w
    sqrt_inv = 1.0 / np.sqrt(rhs)
    A = K * sqrt_inv[:, None] * sqrt_inv[None, :]
    A = (A + A.T) / 2.0  # symmetrize
    vals = np.linalg.eigvalsh(A)
    return vals

def _surrogate_gap(cell, nmax=4):
    """Compute the 4-5 TM gap-midgap ratio on a coarse plane-wave grid."""
    eps_inv, N = _build_eps_grid(cell)
    coeffs = _fourier_coeffs(eps_inv, N, nmax)
    # Irreducible BZ for square lattice: kx in [0, pi], ky in [0, kx]
    # Sample points on the boundary and interior
    kpts = []
    # Gamma-M edge: (k, 0)
    for k in np.linspace(0, np.pi, 8):
        kpts.append((k, 0.0))
    # M-X edge: (pi, k) -> in irreducible, this is (pi, k) for k in [0, pi]
    for k in np.linspace(0, np.pi, 8):
        kpts.append((np.pi, k))
    # X-Gamma edge: (k, k)
    for k in np.linspace(0, np.pi / 2, 6):
        kpts.append((k, k))
    # Interior
    kpts.append((np.pi / 2, np.pi / 4))
    kpts.append((np.pi * 0.75, np.pi * 0.25))
    kpts.append((np.pi * 0.6, np.pi * 0.4))

    max_4 = 0.0
    min_5 = np.inf
    for (kx, ky) in kpts:
        vals = _pw_solve(coeffs, kx, ky, nmax)
        # TM bands: sort, take 4th and 5th (1-indexed)
        if len(vals) >= 5:
            w4 = vals[3]
            w5 = vals[4]
            max_4 = max(max_4, w4)
            min_5 = min(min_5, w5)
    if min_5 == np.inf or max_4 <= 0:
        return -1.0
    gap = min_5 - max_4
    midgap = (min_5 + max_4) / 2.0
    if midgap <= 0:
        return -1.0
    return gap / midgap

# --- Cell construction --------------------------------------------------------
def _make_cell(size, params):
    """Build a symmetric cell from four-rod parameters.
    params: (radius, dx, dy) where rods are at (±dx, ±dy) in cell-normalized coords."""
    radius, dx, dy = params
    cell = np.zeros((size, size), dtype=np.float64)
    for sx in (1, -1):
        for sy in (1, -1):
            cx = 0.5 + sx * dx
            cy = 0.5 + sy * dy
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size
                    y = (j + 0.5) / size
                    if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                        cell[i, j] = 1.0
    return cell

def _make_cell_fast(size, params):
    """Vectorized version for speed."""
    radius, dx, dy = params
    x = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x, x, indexing='ij')
    cell = np.zeros((size, size), dtype=np.float64)
    for sx in (1, -1):
        for sy in (1, -1):
            cx = 0.5 + sx * dx
            cy = 0.5 + sy * dy
            cell += ((X - cx) ** 2 + (Y - cy) ** 2 < radius ** 2).astype(np.float64)
    return (cell > 0).astype(np.float64)

# --- Search -------------------------------------------------------------------
def design(problem):
    size = problem["pixels"]
    # Start with a good four-rod configuration
    best_params = (0.18, 0.25, 0.25)
    best_score = -1.0

    # Evaluate initial
    cell = _make_cell_fast(size, best_params)
    best_score = _surrogate_gap(cell)
    best_cell = cell.copy()

    if time.time() > _DEADLINE:
        return best_cell.astype(int).tolist()

    # Coarse grid search over radius and displacement
    radii = np.linspace(0.10, 0.28, 8)
    dvals = np.linspace(0.18, 0.32, 6)

    for r in radii:
        for d in dvals:
            if time.time() > _DEADLINE - 5:
                break
            params = (r, d, d)
            cell = _make_cell_fast(size, params)
            sc = _surrogate_gap(cell)
            if sc > best_score:
                best_score = sc
                best_params = params
                best_cell = cell.copy()
        if time.time() > _DEADLINE - 5:
            break

    # Local refinement around best
    if time.time() < _DEADLINE - 10:
        r0, d0, _ = best_params
        for dr in np.linspace(-0.03, 0.03, 5):
            for dd in np.linspace(-0.03, 0.03, 5):
                if time.time() > _DEADLINE - 5:
                    break
                r = r0 + dr
                d = d0 + dd
                if r < 0.05 or r > 0.35 or d < 0.1 or d > 0.38:
                    continue
                params = (r, d, d)
                cell = _make_cell_fast(size, params)
                sc = _surrogate_gap(cell)
                if sc > best_score:
                    best_score = sc
                    best_params = params
                    best_cell = cell.copy()

    # Final: build the 32x32 cell and return
    final_cell = _make_cell_fast(size, best_params)
    # Ensure strict symmetry (should already hold, but enforce)
    final_cell = (final_cell + final_cell[::-1, ::-1]) / 2.0
    final_cell = (final_cell + final_cell.T) / 2.0
    final_cell = (final_cell > 0.5).astype(int)

    return final_cell.tolist()
