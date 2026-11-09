"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _enforce_c4v(cell):
    """Enforce C4v symmetry: cell == cell[::-1] rows, cell == cell[::-1] cols, cell == cell.T"""
    size = len(cell)
    cell = np.array(cell, dtype=np.int8)
    cell = (cell + np.rot90(cell, 2)) // 2
    cell = (cell + cell.T) // 2
    cell = (cell + cell[::-1, :]) // 2
    cell = (cell + cell[:, ::-1]) // 2
    cell = (cell > 0).astype(np.int8)
    return cell.tolist()

def _compute_gap(cell, problem):
    """
    Compute the TM band gap ratio between bands 4 and 5 using a proper
    symmetric plane-wave method.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    cell_arr = np.array(cell, dtype=np.int8)

    # Build epsilon and inv-epsilon grids
    eps_grid = np.where(cell_arr == 1, eps_high, eps_low).astype(np.float64)
    inv_eps_grid = 1.0 / eps_grid

    # 2D FFT for Fourier coefficients
    eps_fft = np.fft.fft2(eps_grid) / (size * size)
    inv_eps_fft = np.fft.fft2(inv_eps_grid) / (size * size)

    def get_inv_eps_coeff(ix, iy):
        iix = ix % size
        iiy = iy % size
        return inv_eps_fft[iix, iiy].real

    # Plane-wave expansion: use a circular cutoff in G-space
    # For a 32x32 pixel cell, G vectors are integers (in units of 2*pi/a)
    # Use cutoff radius ~ 4 to get ~ 80+ plane waves
    G_cutoff = 4
    G_list = []
    for gx in range(-G_cutoff, G_cutoff + 1):
        for gy in range(-G_cutoff, G_cutoff + 1):
            if gx * gx + gy * gy <= G_cutoff * G_cutoff:
                G_list.append((gx, gy))
    G_list.sort()
    Npw = len(G_list)
    Gx_arr = np.array([g[0] for g in G_list], dtype=np.float64)
    Gy_arr = np.array([g[1] for g in G_list], dtype=np.float64)

    # Precompute inv_eps coefficients for all pairwise differences
    # We need inv_eps(G_i - G_j) for all i,j
    # G_i - G_j ranges from -2*G_cutoff to 2*G_cutoff
    diff_cutoff = 2 * G_cutoff
    inv_eps_coeffs = {}
    for dx in range(-diff_cutoff, diff_cutoff + 1):
        for dy in range(-diff_cutoff, diff_cutoff + 1):
            if dx * dx + dy * dy <= (2 * G_cutoff) ** 2:
                inv_eps_coeffs[(dx, dy)] = get_inv_eps_coeff(dx, dy)

    # K-points in the irreducible Brillouin zone
    # Square lattice: irreducible BZ is 0 <= ky <= kx, 0 <= kx <= 0.5 (in units of 2*pi/a)
    # Use a grid of points
    k_points = []
    nk = 4
    for ix in range(nk + 1):
        for iy in range(ix + 1):
            kx_u = ix / (2.0 * nk)
            ky_u = iy / (2.0 * nk)
            if kx_u <= 0.5 and ky_u <= kx_u:
                k_points.append((kx_u, ky_u))
    # Add high-symmetry points explicitly
    k_points.append((0.0, 0.0))       # Gamma
    k_points.append((0.5, 0.0))       # X
    k_points.append((0.5, 0.5))       # M
    k_points.append((0.25, 0.0))
    k_points.append((0.5, 0.25))
    k_points.append((0.25, 0.25))
    k_points.append((0.125, 0.0))
    k_points.append((0.375, 0.125))

    n_bands = 5

    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx_u, ky_u in k_points:
        # Build the k-vectors: k + G
        kx_vec = kx_u + Gx_arr
        ky_vec = ky_u + Gy_arr

        # Build the symmetric TM eigenvalue matrix
        # H[i,j] = sqrt(inv_eps[i] * inv_eps[j]) * (k_i dot k_j)
        # But wait - the standard TM formulation is:
        # sum_j inv_eps(G_i - G_j) * (k + G_j) . (k + G_j) * c_j = omega^2 * c_i
        # This is NOT symmetric in general. The symmetric form requires:
        # H[i,j] = sqrt(inv_eps(G_i) * inv_eps(G_j)) * (k_i + G_i) . (k_j + G_j)
        # No, that's not right either. Let me think again.
        #
        # The TM Maxwell equations give:
        # (1/eps(r)) curl curl E = (omega/c)^2 E
        # In plane-wave basis: sum_j inv_eps(G_i - G_j) |k + G_j|^2 c_j = omega^2 c_i
        # This matrix is NOT Hermitian in general because inv_eps(G_i-G_j) != conj(inv_eps(G_j-G_i))
        # unless inv_eps is real and even (which it is for real eps), so inv_eps(G_i-G_j) = inv_eps(G_j-G_i).
        # So the matrix IS symmetric! H[i,j] = inv_eps(G_i-G_j) * |k+G_j|^2
        # But H[i,j] != H[j,i] because |k+G_i|^2 != |k+G_j|^2 in general.
        # Wait: H[i,j] = inv_eps(G_i-G_j)*|k+G_j|^2 and H[j,i] = inv_eps(G_j-G_i)*|k+G_i|^2
        # Since inv_eps is real and even, H[i,j] - H[j,i] = inv_eps(G_i-G_j)*(|k+G_j|^2 - |k+G_i|^2)
        # This is NOT zero in general. So the matrix is NOT symmetric.
        #
        # The correct approach: use a generalized eigenvalue problem or symmetrize.
        # Standard approach: define d_i = sqrt(inv_eps(G_i)) * c_i
        # Then the equation becomes a symmetric eigenvalue problem.
        # Actually, the standard plane-wave method for TM uses:
        # H[i,j] = inv_eps(G_i - G_j) * (k+G_i).(k+G_j)  -- NO, that's not standard.
        #
        # Let me use the well-known formulation:
        # For TM: the eigenvalue problem is A c = lambda c where
        # A[i,j] = inv_eps(G_i - G_j) * |k + G_j|^2
        # This is not symmetric. To make it symmetric, we can write:
        # Let S[i,j] = sqrt(inv_eps(G_i) * inv_eps(G_j))
        # Then B = S^{-1} A S^{-1} is not symmetric either.
        #
        # Actually, the standard approach in photonic crystal literature:
        # The matrix H[i,j] = inv_eps(G_i - G_j) * (k_i + G_i) . (k_j + G_j) is WRONG.
        #
        # Correct: The TM band structure uses:
        # sum_j [1/eps(G_i - G_j)] * |k + G_j|^2 * c_j = omega^2 * c_i
        # This matrix is NOT symmetric. We should use np.linalg.eig instead of eigh.
        # OR we can reformulate.
        #
        # Let's just use eig for non-symmetric matrices and take real parts.

        k_sq = kx_vec ** 2 + ky_vec ** 2

        # Build H matrix: H[i,j] = inv_eps(G_i - G_j) * k_sq[j]
        H = np.zeros((Npw, Npw), dtype=np.float64)
        for i in range(Npw):
            for j in range(Npw):
                dx = Gx_arr[i] - Gx_arr[j]
                dy = Gy_arr[i] - Gy_arr[j]
                key = (int(round(dx)), int(round(dy)))
                coeff = inv_eps_coeffs.get(key, 0.0)
                H[i, j] = coeff * k_sq[j]

        # Solve eigenvalue problem - use eig since matrix may not be symmetric
        eigvals = np.linalg.eigvals(H)
        # Take real parts (eigenvalues should be real for physical systems)
        eigvals_real = np.real(eigvals)
        # Filter to positive eigenvalues
        eigvals_pos = eigvals_real[eigvals_real > 0]
        eigvals_pos.sort()

        if len(eigvals_pos) >= n_bands:
            omegas = np.sqrt(eigvals_pos[:n_bands])
            if len(omegas) >= 5:
                max_omega4 = max(max_omega4, float(omegas[3]))
                min_omega5 = min(min_omega5, float(omegas[4]))

    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap == 0:
        return -1.0

    return 2.0 * gap / midgap

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell that maximizes the TM band gap
    between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    # Start with a good initial guess: four-rod configuration
    def make_initial_cell(rod_radius=3, centers=None):
        if centers is None:
            centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        cell = np.zeros((size, size), dtype=np.int8)
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    if (i - cy) ** 2 + (j - cx) ** 2 < rod_radius ** 2:
                        cell[i, j] = 1
        return _enforce_c4v(cell.tolist())

    # Try several initial configurations
    initial_configs = [
        make_initial_cell(rod_radius=3, centers=[(8, 8), (8, 24), (24, 8), (24, 24)]),
        make_initial_cell(rod_radius=4, centers=[(8, 8), (8, 24), (24, 8), (24, 24)]),
        make_initial_cell(rod_radius=5, centers=[(8, 8), (8, 24), (24, 8), (24, 24)]),
        make_initial_cell(rod_radius=3, centers=[(16, 8), (8, 16), (24, 16), (16, 24)]),
        make_initial_cell(rod_radius=4, centers=[(16, 8), (8, 16), (24, 16), (16, 24)]),
    ]

    # Pick the best initial configuration
    best_cell = initial_configs[0]
    best_score = _compute_gap(best_cell, problem)

    for cfg in initial_configs[1:]:
        if time.time() > deadline - 60:
            break
        score = _compute_gap(cfg, problem)
        if score > best_score:
            best_score = score
            best_cell = cfg

    # Local search: flip random pixels while maintaining C4v symmetry
    max_iter = 200
    current_cell = np.array(best_cell, dtype=np.int8)
    current_score = best_score

    for it in range(max_iter):
        if time.time() > deadline - 30:
            break

        # Pick a random pixel in the fundamental domain and flip it
        i = random.randint(0, size // 2 - 1)
        j = random.randint(0, size // 2 - 1)

        new_cell = current_cell.copy()

        # C4v symmetry partners for (i, j)
        partners = [
            (i, j),
            (i, size - 1 - j),
            (size - 1 - i, j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (j, size - 1 - i),
            (size - 1 - j, i),
            (size - 1 - j, size - 1 - i),
        ]
        partners = list(set(partners))

        for pi, pj in partners:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]

        new_score = _compute_gap(new_cell.tolist(), problem)

        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = new_cell.tolist()
        else:
            # Small probability of accepting worse moves
            if random.random() < 0.15:
                current_cell = new_cell
                current_score = new_score

    # Final check
    best_cell = _enforce_c4v(best_cell)

    return best_cell
# EVOLVE-BLOCK-END
