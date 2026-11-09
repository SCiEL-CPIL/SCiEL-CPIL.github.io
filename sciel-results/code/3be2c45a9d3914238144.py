"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """Local search over 4-fold symmetric 32x32 cells to maximize TM band-4/5 gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    deadline = time.time() + 210.0  # safety margin before 230s wall-clock limit

    # Precompute plane-wave vectors (289 = 17x17 grid, |G| <= 8.5)
    N = 17
    half = N // 2  # 8
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            if gx * gx + gy * gy <= (half + 0.5) ** 2:  # |G| <= 8.5
                G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors, dtype=float)
    n_pw = len(G_vectors)

    # Precompute 30 k-points in irreducible BZ
    k_points = []
    # Along Gamma-X: kx from 0 to 0.5, ky=0
    for i in range(11):
        k_points.append((i * 0.5 / 10.0, 0.0))
    # Along X-M: kx=0.5, ky from 0 to 0.5
    for i in range(1, 11):
        k_points.append((0.5, i * 0.5 / 10.0))
    # Along M-Gamma: kx from 0.5 to 0, ky from 0.5 to 0
    for i in range(1, 11):
        t = i / 10.0
        k_points.append((0.5 * (1 - t), 0.5 * (1 - t)))
    # Remove duplicates
    unique_k = []
    for k in k_points:
        if not any(abs(k[0] - uk[0]) < 1e-9 and abs(k[1] - uk[1]) < 1e-9 for uk in unique_k):
            unique_k.append(k)
    k_points = np.array(unique_k[:30], dtype=float)

    # Precompute coefficient index matrix
    coeff_lookup = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        coeff_lookup[(int(gx), int(gy))] = idx

    coeff_idx_mat = np.full((n_pw, n_pw), -1, dtype=int)
    for i in range(n_pw):
        for j in range(n_pw):
            dx = int(G_vectors[i, 0] - G_vectors[j, 0])
            dy = int(G_vectors[i, 1] - G_vectors[j, 1])
            key = (dx, dy)
            if key in coeff_lookup:
                coeff_idx_mat[i, j] = coeff_lookup[key]

    valid_mask = coeff_idx_mat >= 0

    def compute_inv_eps_fourier(cell_arr):
        """Compute Fourier coefficients of 1/eps for the given cell."""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_inv = np.fft.fft2(inv_eps)
        c = fft_inv / (size * size)
        coeffs = np.zeros(n_pw, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            ix = int(gx) % size
            iy = int(gy) % size
            coeffs[idx] = c[iy, ix]
        return coeffs

    def solve_bands(cell_arr):
        """Solve for TM bands at all k-points, return (max_omega_4, min_omega_5)."""
        coeffs = compute_inv_eps_fourier(cell_arr)
        coeffs_real = np.real(coeffs)

        max_w4 = 0.0
        min_w5 = float('inf')

        for kx, ky in k_points:
            kG_x = kx + G_vectors[:, 0]
            kG_y = ky + G_vectors[:, 1]

            # Build cross product matrix
            cross = np.outer(kG_x, kG_y) - np.outer(kG_y, kG_x)

            # Build coefficient matrix using precomputed indices
            C = np.zeros((n_pw, n_pw), dtype=float)
            C[valid_mask] = coeffs_real[coeff_idx_mat[valid_mask]]

            M = C * cross
            M = (M + M.T) / 2.0

            eigvals = np.linalg.eigvalsh(M)

            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0, eigvals[3]))
                w5 = math.sqrt(max(0, eigvals[4]))
                max_w4 = max(max_w4, w4)
                min_w5 = min(min_w5, w5)

        return max_w4, min_w5

    def score_cell(cell_arr):
        """Compute the gap-midgap ratio."""
        max_w4, min_w5 = solve_bands(cell_arr)
        if min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)

    # Initialize with the high-scoring inspiration design:
    # Dielectric background (1) with 4 air holes at (+/-0.25, +/-0.25)
    cell_arr = np.ones((size, size), dtype=int)
    r = 0.28
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    cell_arr[i, j] = 0
                    break

    best_score = score_cell(cell_arr)
    best_cell = cell_arr.copy()

    # Local search: flip pixels while maintaining 4-fold symmetry
    # A pixel at (i,j) is linked to (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
    # Also (j,i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i) for transpose
    
    rng = np.random.default_rng(42)

    # Precompute symmetry orbits under the full square symmetry group
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                # Generate all symmetric equivalents
                orbit = set()
                orbit.add((i, j))
                orbit.add((j, i))  # transpose
                orbit.add((size - 1 - i, j))  # vertical flip
                orbit.add((i, size - 1 - j))  # horizontal flip
                orbit.add((size - 1 - i, size - 1 - j))  # 180 rot
                orbit.add((j, size - 1 - i))  # transpose + vflip
                orbit.add((size - 1 - j, i))  # transpose + hflip
                orbit.add((size - 1 - j, size - 1 - i))  # transpose + 180 rot
                
                unique_orbit = []
                for p in orbit:
                    if not visited[p[0], p[1]]:
                        unique_orbit.append(p)
                        visited[p[0], p[1]] = True
                if unique_orbit:
                    orbits.append(unique_orbit)

    # Sort orbits by distance from center for better search
    orbits.sort(key=lambda orb: min(math.sqrt((i - 15.5)**2 + (j - 15.5)**2) for i, j in orb))

    iteration = 0
    while time.time() < deadline - 5.0:
        # Pick a random orbit
        idx = rng.integers(0, len(orbits))
        orbit = orbits[idx]

        # Flip all pixels in the orbit
        for (i, j) in orbit:
            cell_arr[i, j] = 1 - cell_arr[i, j]

        new_score = score_cell(cell_arr)

        if new_score > best_score:
            best_score = new_score
            best_cell = cell_arr.copy()
        else:
            # Revert
            for (i, j) in orbit:
                cell_arr[i, j] = 1 - cell_arr[i, j]

        iteration += 1

    # Return best cell as nested list
    return [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
