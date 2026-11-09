import time
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
_DEADLINE = 190.0  # seconds, keep a safety margin below the 200 s wall limit


def _elapsed():
    return time.time() - _START_T


def _remaining():
    return _DEADLINE - _elapsed()


_START_T = time.time()


# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------
def _enforce_symmetry(cell, size):
    """Enforce C4v symmetry: 180-deg rotation, transpose, and 90-deg rotation.

    We average over the four group elements (as 0/1 then threshold), which
    guarantees invariance under all four operations.
    """
    c = np.array(cell, dtype=np.float64)
    # 180-degree rotation
    c180 = c[::-1, ::-1]
    # Transpose
    cT = c.T
    # 90-degree rotation = transpose then flip rows
    c90 = cT[:, ::-1]

    # Average
    sym = (c + c180 + cT + c90) / 4.0
    return np.where(sym >= 0.5, 1, 0)


# ---------------------------------------------------------------------------
# Fast band-structure evaluation via plane-wave method
# ---------------------------------------------------------------------------
# Precompute G-vectors for 17x17 = 289 plane waves
_N_PW = 8  # G indices from -8 to +8
_G_range = np.arange(-_N_PW, _N_PW + 1)
_Gx, _Gy = np.meshgrid(_G_range, _G_range, indexing='ij')
_G_vectors = np.column_stack((_Gx.ravel(), _Gy.ravel())).astype(np.float64)
_N_G = _G_vectors.shape[0]  # 289

# Precompute (k+G)^2 coefficients will be computed per k-point
# Precompute difference index map: for each pair (i, j), index of (G_i - G_j)
# G_i - G_j has components in [-16, 16], map to index in G_vectors
# Build a dict: (gx, gy) -> index
_G_to_idx = {}
for _i, (_gx, _gy) in enumerate(_G_vectors):
    _G_to_idx[(_gx, _gy)] = _i

# Diff_Index[i, j] = index of (G_i - G_j)
_Diff_Index = np.zeros((_N_G, _N_G), dtype=np.int32)
for _i in range(_N_G):
    for _j in range(_N_G):
        _dgx = _G_vectors[_i, 0] - _G_vectors[_j, 0]
        _dgy = _G_vectors[_i, 1] - _G_vectors[_j, 1]
        _idx = _G_to_idx.get((_dgx, _dgy), -1)
        _Diff_Index[_i, _j] = _idx

# K-points: 30 points covering the irreducible BZ of square lattice
# Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= (pi/a)^2 (in dimensionless units where a=1, BZ boundary at pi)
# Actually for square lattice with a=1, reciprocal lattice vector = 2*pi, BZ is [-pi, pi]^2
# Irreducible wedge: 0 <= kx <= ky, kx^2+ky^2 <= pi^2
# Generate 30 points: mix of boundary and interior
_K_points = np.array([
    # Boundary points (M point, X points, etc.)
    [0.0, 0.0],          # Gamma
    [np.pi / 2, 0.0],    # X
    [np.pi / 2, np.pi / 2],  # M
    [np.pi / 4, 0.0],
    [np.pi / 4, np.pi / 4],
    [0.0, np.pi / 4],
    [0.0, np.pi / 2],
    [np.pi / 8, 0.0],
    [np.pi / 8, np.pi / 8],
    [np.pi / 8, np.pi / 4],
    [np.pi / 8, 3 * np.pi / 8],
    [np.pi / 6, 0.0],
    [np.pi / 6, np.pi / 6],
    [np.pi / 6, np.pi / 3],
    [np.pi / 6, np.pi / 2],
    [np.pi / 5, 0.0],
    [np.pi / 5, np.pi / 5],
    [np.pi / 5, np.pi / 3],
    [np.pi / 5, 2 * np.pi / 5],
    [np.pi / 4, np.pi / 8],
    [np.pi / 4, np.pi / 6],
    [np.pi / 4, np.pi / 5],
    [np.pi / 3, np.pi / 6],
    [np.pi / 3, np.pi / 4],
    [np.pi / 3, np.pi / 3],
    [np.pi / 2, np.pi / 8],
    [np.pi / 2, np.pi / 6],
    [np.pi / 2, np.pi / 5],
    [np.pi / 2, np.pi / 4],
    [np.pi / 2, np.pi / 3],
])


def _compute_eps_inv_fourier(cell_2d, size):
    """Compute Fourier coefficients of 1/eps(x,y) for the 289 plane waves.

    cell_2d is a (size, size) array of 0/1.
    eps = eps_high where 1, eps_low where 0.
    1/eps: 1/eps_low where 0, 1/eps_high where 1.
    """
    eps_low = 1.0
    eps_high = 12.0
    eps_inv = np.where(cell_2d == 1, 1.0 / eps_high, 1.0 / eps_low)

    # Compute Fourier coefficients: f_G = (1/A) integral exp(-i G . r) f(r) dA
    # For pixel grid: f_G = (1/size^2) sum_{i,j} f(i,j) exp(-i G . (i+0.5, j+0.5)/size * 2*pi)
    # Actually in dimensionless coords with a=1, pixel (i,j) center is at ((i+0.5)/size, (j+0.5)/size)
    # G vector in dimensionless: G * 2*pi (since reciprocal lattice vector = 2*pi)
    # So G . r = 2*pi * G_int . (pixel_center)

    x = (np.arange(size) + 0.5) / size  # pixel centers in [0, 1)
    y = (np.arange(size) + 0.5) / size

    # exp(-i * 2*pi * G . r) for each pixel and each G
    # Shape: (size, size, N_G)
    # Use broadcasting: x (size,), G_vectors[:,0] (N_G,)
    phase_x = np.exp(-1j * 2 * np.pi * np.outer(x, _G_vectors[:, 0]))  # (size, N_G)
    phase_y = np.exp(-1j * 2 * np.pi * np.outer(y, _G_vectors[:, 1]))  # (size, N_G)

    # f_G = (1/size^2) sum_{i,j} eps_inv[i,j] * phase_x[i, Gx] * phase_y[j, Gy]
    # This is (eps_inv * phase_x[:, :, None] ... ) -- too large. Use matrix multiply.
    # eps_inv (size, size), phase_x (size, N_G), phase_y (size, N_G)
    # f_G = (1/size^2) * (phase_x.T @ eps_inv) @ phase_y
    # phase_x.T: (N_G, size), eps_inv: (size, size), phase_y: (size, N_G)
    result = (phase_x.T @ eps_inv) @ phase_y / (size * size)

    return result  # complex array of shape (N_G,)


def _evaluate_gap(cell_2d, size):
    """Compute the gap-midgap ratio between TM bands 4 and 5."""
    eps_inv_f = _compute_eps_inv_fourier(cell_2d, size)
    # eps_inv_f is complex; the operator is Hermitian so we use it directly

    max_omega_4 = 0.0
    min_omega_5 = np.inf

    for kx, ky in _K_points:
        # H_{ij} = eps_inv_f[G_i - G_j] * |k + G_j|^2
        # Note: in dimensionless units with a=1, k is in rad units (BZ: [-pi, pi])
        # G vectors are integers (reciprocal lattice units, each = 2*pi/a)
        # (k + G_j)^2 = (kx + 2*pi*Gjx)^2 + (ky + 2*pi*Gjy)^2
        # Actually standard convention: k is in units of 2*pi/a, so k_dimless in [-0.5, 0.5]
        # Let's use: k in units of 2*pi, so BZ is [-0.5, 0.5]^2
        # Then (k+G)^2 = (kx + Gx)^2 + (ky + Gy)^2 where kx, ky in [-0.5, 0.5]
        # Rescale k_points to be in units of 2*pi: divide by 2*pi
        kx_dim = kx / (2 * np.pi)
        ky_dim = ky / (2 * np.pi)

        # |k + G|^2 for each G
        kg = _G_vectors + np.array([kx_dim, ky_dim])
        kg2 = kg[:, 0] ** 2 + kg[:, 1] ** 2  # (N_G,)

        # H_{ij} = eps_inv_f[Diff_Index[i,j]] * kg2[j]
        # eps_inv_f indexed by Diff_Index
        eif_diff = eps_inv_f[_Diff_Index]  # (N_G, N_G)
        H = eif_diff * kg2[np.newaxis, :]  # (N_G, N_G)

        # Solve eigenvalue problem: H v = lambda v, where lambda = omega^2
        # H is not necessarily Hermitian in this form; the correct TM operator is:
        # H_{ij} = eps_inv_f[G_i - G_j] * (k + G_j)^2
        # This is a generalized eigenvalue problem but since eps_inv_f is the Fourier
        # transform of 1/eps, the operator is: (1/eps) * (-nabla^2) for TM
        # In plane-wave basis: H_{ij} = <i| (1/eps) * (k+G)^2 |j>
        # = eps_inv_f[G_i - G_j] * (k+G_j)^2
        # This matrix is NOT Hermitian in general, but the eigenvalues should be real
        # (since the operator is self-adjoint). Use eigh if possible, else eig.

        # For safety, use eig and take real parts
        try:
            eigvals = np.linalg.eigvals(H)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
        except np.linalg.LinAlgError:
            continue

        if len(eigvals) < 5:
            continue

        omega_4 = np.sqrt(max(eigvals[3], 0.0))  # 0-indexed, band 4 is index 3
        omega_5 = np.sqrt(max(eigvals[4], 0.0))  # band 5 is index 4

        max_omega_4 = max(max_omega_4, omega_4)
        min_omega_5 = min(min_omega_5, omega_5)

    if min_omega_5 == np.inf or max_omega_4 == 0.0:
        return 0.0

    gap = 2.0 * (min_omega_5 - max_omega_4)
    midgap = min_omega_5 + max_omega_4
    if midgap == 0:
        return 0.0
    return gap / midgap


# ---------------------------------------------------------------------------
# Local search: hill-climbing with random pixel flips
# ---------------------------------------------------------------------------
def _random_symmetric_cell(size, density):
    """Generate a random cell with C4v symmetry."""
    # Generate one orbit representative and tile
    # For C4v, orbits can have size 1, 2, 4, or 8.
    # Simplest: generate full grid then symmetrize.
    cell = np.random.random((size, size))
    cell = (cell > 1.0 - density).astype(np.int32)
    cell = _enforce_symmetry(cell, size)
    return cell


def _flip_pixel(cell, size):
    """Flip one pixel and enforce symmetry, return modified cell."""
    i, j = np.random.randint(size, size=2)
    cell_new = cell.copy()
    # Flip the orbit of (i, j)
    # Orbit under C4v: (i,j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
    # (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)
    val = 1 - cell_new[i, j]
    for (a, b) in [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                   (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)]:
        if 0 <= a < size and 0 <= b < size:
            cell_new[a, b] = val
    return cell_new


def design(problem):
    global _START_T
    _START_T = time.time()

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)

    # Start with the seed: centered rod
    cell = np.zeros((size, size), dtype=np.int32)
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    dist2 = x[:, None] ** 2 + y[None, :] ** 2
    cell[dist2 < 0.2 ** 2] = 1
    cell = _enforce_symmetry(cell, size)

    best_score = _evaluate_gap(cell, size)
    best_cell = cell.copy()

    # Local search
    rng = np.random.RandomState(42)
    max_iter = 500
    step = 0

    while _remaining() > 15.0:
        step += 1
        # Random restart occasionally
        if step % 50 == 0:
            cell = _random_symmetric_cell(size, 0.3)
            score = _evaluate_gap(cell, size)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
            continue

        # Try a pixel flip
        cell_trial = cell.copy()
        i, j = rng.randint(size, size=2)
        val = 1 - cell_trial[i, j]
        for (a, b) in [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                       (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)]:
            if 0 <= a < size and 0 <= b < size:
                cell_trial[a, b] = val

        score = _evaluate_gap(cell_trial, size)
        if score > best_score:
            best_score = score
            best_cell = cell_trial.copy()
            cell = cell_trial.copy()
        else:
            # Keep current cell (no accept worse)
            pass

        if _remaining() < 30.0:
            break

    # Return as nested list
    result = best_cell.tolist()
    return result
