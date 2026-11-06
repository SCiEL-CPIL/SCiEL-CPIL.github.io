"""Photonic crystal design: 32x32 D4-symmetric unit cell for max TM band-4/5 gap."""
import time
import math
import numpy as np
from scipy import linalg

_DEADLINE = 230.0  # wall-clock seconds; evaluator kills at this point
_TIME_MARGIN = 10.0  # safety margin


def _d4_symmetrize(cell):
    """Enforce D4 symmetry: row-reverse, col-reverse, and transpose invariance."""
    cell = np.array(cell, dtype=np.int32)
    # Average over all 8 D4 operations, then threshold
    ops = [
        cell,
        cell[::-1, :],
        cell[:, ::-1],
        cell[::-1, ::-1],
        cell.T,
        cell.T[::-1, :],
        cell.T[:, ::-1],
        cell.T[::-1, ::-1],
    ]
    sym = np.zeros_like(cell, dtype=np.float64)
    for op in ops:
        sym += op
    sym /= 8.0
    return (sym >= 0.5).astype(np.int32)


def _compute_gap(cell, size=32, n_pw=17, n_bands=5, n_kpts=10):
    """
    Compute the TM band-4/5 gap-midgap ratio using plane-wave expansion.
    cell: 32x32 binary array (1=dielectric eps=12, 0=air eps=1)
    """
    eps_high = 12.0
    eps_low = 1.0

    # Build epsilon grid
    eps_grid = np.where(cell.astype(np.float64) == 1, eps_high, eps_low)

    # Fourier coefficients of 1/epsilon for TM mode
    # We need 1/eps for TM: H_z equation involves 1/eps
    inv_eps_grid = 1.0 / eps_grid

    # Plane wave indices: G = (m, n) * 2*pi, |m|,|n| <= n_pw
    # Use hexagonal truncation for efficiency: m^2 + n^2 <= (n_pw+0.5)^2
    G_list = []
    for m in range(-n_pw, n_pw + 1):
        for n in range(-n_pw, n_pw + 1):
            if m * m + n * n <= (n_pw + 0.5) ** 2:
                G_list.append((m, n))
    N = len(G_list)
    G_arr = np.array(G_list, dtype=np.float64) * 2.0 * np.pi

    # Precompute Fourier coefficients of 1/eps
    # c_G = (1/Npix) * sum_{i,j} inv_eps[i,j] * exp(-i * G . r_ij)
    # r_ij = ((i+0.5)/size, (j+0.5)/size) - 0.5
    i_idx, j_idx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x = (i_idx + 0.5) / size - 0.5
    y = (j_idx + 0.5) / size - 0.5
    r_flat = np.stack([x.ravel(), y.ravel()], axis=1)  # (size^2, 2)

    # Phase matrix: (size^2, N)
    # phase[k, idx] = -2*pi * (m_k * x_k + n_k * y_k)
    G_flat = G_arr  # (N, 2)
    phase = -2.0 * np.pi * (r_flat[:, 0:1] * G_flat[0, :][np.newaxis, :] +
                            r_flat[:, 1:2] * G_flat[1, :][np.newaxis, :])
    # Actually: phase = -2*pi*(m*x + n*y) for each G and each pixel
    # r_flat[:,0] is x, r_flat[:,1] is y
    # G_flat[0,:] is m values, G_flat[1,:] is n values
    phase = -(r_flat[:, 0:1] * G_flat[0, :][np.newaxis, :] +
              r_flat[:, 1:2] * G_flat[1, :][np.newaxis, :]) * 2.0 * np.pi

    # c_G = mean over pixels of inv_eps * exp(i * phase)  [sign convention]
    exp_phase = np.exp(1j * phase)
    c_G = np.mean(inv_eps_grid.ravel() * exp_phase, axis=0)  # (N,)

    # For each k-point, build the matrix:
    # For TM: (k+G)^2 * c_{G'-G}  ... actually the eigenvalue problem is:
    # sum_{G'} c_{G-G'} * |k+G'|^2 * a_{G'} = omega^2 * a_G
    # where c_G are Fourier coeffs of 1/eps
    # So M[G, G'] = c_{G-G'} * |k+G'|^2
    # We need c_{G-G'} for all pairs.

    # Build c_diff matrix: c_diff[i,j] = c_{G_i - G_j}
    # G_i - G_j corresponds to index (m_i - m_j, n_i - n_j) in G_list
    # Map (m,n) -> index
    G_map = {}
    for idx, (m, n) in enumerate(G_list):
        G_map[(m, n)] = idx

    c_diff = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        mi, ni = G_list[i]
        for j in range(N):
            mj, nj = G_list[j]
            key = (mi - mj, ni - nj)
            if key in G_map:
                c_diff[i, j] = c_G[G_map[key]]
            else:
                c_diff[i, j] = 0.0  # outside truncation

    # K-points in irreducible BZ: from (0,0) to (pi,0) to (pi,pi) to (0,pi)
    # Irreducible: 0 <= kx <= ky, kx+ky <= pi (for square lattice with 2pi periodicity)
    # Actually for square lattice with lattice constant a=1, BZ is [-pi,pi]^2
    # Irreducible: 0 <= kx <= ky, kx + ky <= pi
    kpts = []
    # Along kx=0: ky from 0 to pi
    for t in np.linspace(0, np.pi, n_kpts):
        kpts.append((0.0, t))
    # Along ky=kx: from (0,0) to (pi/2, pi/2)
    for t in np.linspace(0, np.pi / 2, max(1, n_kpts // 2)):
        kpts.append((t, t))
    # Along ky=pi: kx from 0 to pi/2 (but kx+ky<=pi means kx<=0 when ky=pi, so just (0,pi))
    # Actually the boundary is: kx=0, ky in [0,pi]; ky=kx, kx in [0, pi/2]; and the line kx+ky=pi
    # Let me redo: irreducible BZ for square: 0 <= kx <= ky, kx + ky <= pi
    # Boundaries: (1) kx=0, 0<=ky<=pi; (2) ky=kx, 0<=kx<=pi/2; (3) kx+ky=pi, 0<=kx<=pi/2
    kpts = []
    for t in np.linspace(0, np.pi, n_kpts):
        kpts.append((0.0, t))
    for t in np.linspace(0, np.pi / 2, max(1, n_kpts // 2)):
        kpts.append((t, t))
    for t in np.linspace(0, np.pi / 2, max(1, n_kpts // 2)):
        kpts.append((t, np.pi - t))

    max_omega4 = 0.0
    min_omega5 = np.inf

    for kx, ky in kpts:
        k = np.array([kx, ky])
        # |k+G|^2 for each G
        kg = G_arr + k[np.newaxis, :]
        kg2 = np.sum(kg ** 2, axis=1)  # (N,)

        # Matrix M[i,j] = c_diff[i,j] * kg2[j]
        M = c_diff * kg2[np.newaxis, :]
        # M should be Hermitian: M[i,j] = c_{G_i-G_j} * |k+G_j|^2
        # c_{-G} = conj(c_G) for real eps, so M is Hermitian
        M = (M + M.conj().T) / 2.0  # enforce Hermiticity

        # Solve eigenvalue problem
        try:
            eigvals = linalg.eigvalsh(M)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                # omega = sqrt(eigenvalue) since we set up the problem with omega^2
                omega4 = math.sqrt(max(eigvals[3], 0.0))
                omega5 = math.sqrt(max(eigvals[4], 0.0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except Exception:
            continue

    if min_omega5 == np.inf or max_omega4 == 0:
        return -1.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap


def design(problem):
    """Design a 32x32 D4-symmetric photonic crystal unit cell maximizing TM band-4/5 gap."""
    size = problem.get("pixels", 32)
    t0 = time.time()

    # Start with a 2x2 array of small rods (measured improvement over single rod)
    # This creates a sub-lattice that enhances Bragg scattering at higher frequencies

    def _make_rod_array(n_rods=2, rod_radius=0.15, offset=0.0):
        """Create n_rods x n_rods array of circular rods."""
        cell = np.zeros((size, size), dtype=np.float64)
        # Place rods in a grid
        spacing = 1.0 / n_rods
        for ii in range(n_rods):
            for jj in range(n_rods):
                cx = (ii + 0.5) * spacing - 0.5 + offset
                cy = (jj + 0.5) * spacing - 0.5 + offset
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        if (x - cx) ** 2 + (y - cy) ** 2 < rod_radius ** 2:
                            cell[i, j] = 1.0
        return cell

    # Initial candidate: 2x2 rod array
    best_cell = _make_rod_array(n_rods=2, rod_radius=0.15)
    best_cell = _d4_symmetrize(best_cell)
    best_score = _compute_gap(best_cell)

    # Local search: perturb small regions and check improvement
    t_start = time.time()
    n_iter = 0
    max_iter = 200

    # Define perturbation: flip a small D4-symmetric patch
    def _d4_flip(cell, center_i, center_j, radius):
        """Flip pixels in a D4-symmetric pattern around (center_i, center_j)."""
        new_cell = cell.copy()
        # Get the 8 D4-equivalent positions
        ops = [
            (center_i, center_j),
            (size - 1 - center_i, center_j),
            (center_i, size - 1 - center_j),
            (size - 1 - center_i, size - 1 - center_j),
            (center_j, center_i),
            (center_j, size - 1 - center_i),
            (size - 1 - center_j, center_i),
            (size - 1 - center_j, size - 1 - center_i),
        ]
        for (ci, cj) in ops:
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if di * di + dj * dj <= radius * radius:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            new_cell[ni, nj] = 1 - new_cell[ni, nj]
        return new_cell

    # Random D4-symmetric perturbations
    rng = np.random.RandomState(42)
    for iteration in range(max_iter):
        if time.time() - t0 > _DEADLINE - _TIME_MARGIN:
            break

        # Choose a random center and radius for perturbation
        # Focus on the first quadrant for D4 symmetry
        ci = rng.randint(0, size // 2)
        cj = rng.randint(ci, size // 2)  # ensure ci <= cj for irreducible
        radius = rng.choice([1, 2, 3])

        candidate = _d4_flip(best_cell, ci, cj, radius)
        candidate = _d4_symmetrize(candidate)

        score = _compute_gap(candidate)
        n_iter += 1

        if score > best_score:
            best_score = score
            best_cell = candidate

    # Final: ensure D4 symmetry and return as nested list
    final_cell = _d4_symmetrize(best_cell)
    return [[int(final_cell[i][j]) for j in range(size)] for i in range(size)]
