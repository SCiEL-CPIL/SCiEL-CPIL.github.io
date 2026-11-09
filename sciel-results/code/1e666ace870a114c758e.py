"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _enforce_c4v(cell):
    """Enforce C4v symmetry: cell == cell[::-1] rows, cell == cell[::-1] cols, cell == cell.T"""
    size = len(cell)
    cell = np.array(cell, dtype=np.int8)
    # Enforce all symmetry operations
    cell = (cell + np.rot90(cell, 2)) // 2
    cell = (cell + cell.T) // 2
    cell = (cell + cell[::-1, :]) // 2
    cell = (cell + cell[:, ::-1]) // 2
    cell = (cell > 0.5).astype(np.int8)
    return cell.tolist()

def _compute_gap_fast(cell, problem):
    """
    Fast computation of TM band gap between bands 4 and 5.
    Uses a reduced set of k-points and plane waves for speed.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    cell_arr = np.array(cell, dtype=np.int8)
    eps_grid = np.where(cell_arr == 1, eps_high, eps_low).astype(np.float64)
    inv_eps_grid = 1.0 / eps_grid

    # FFT of inv_eps
    inv_eps_fft = np.fft.fft2(inv_eps_grid) / (size * size)

    def get_coeff(ix, iy):
        return inv_eps_fft[ix % size, iy % size].real

    # Plane wave cutoff - use radius 3 for speed (49 plane waves)
    G_cutoff = 3
    G_list = []
    for gx in range(-G_cutoff, G_cutoff + 1):
        for gy in range(-G_cutoff, G_cutoff + 1):
            if gx * gx + gy * gy <= G_cutoff * G_cutoff:
                G_list.append((gx, gy))
    G_list.sort()
    Npw = len(G_list)
    Gx_arr = np.array([g[0] for g in G_list], dtype=np.float64)
    Gy_arr = np.array([g[1] for g in G_list], dtype=np.float64)

    # Precompute inv_eps coefficients for differences
    diff_cutoff = 2 * G_cutoff
    coeff_cache = {}
    for dx in range(-diff_cutoff, diff_cutoff + 1):
        for dy in range(-diff_cutoff, diff_cutoff + 1):
            if dx * dx + dy * dy <= (2 * G_cutoff) ** 2:
                coeff_cache[(dx, dy)] = get_coeff(dx, dy)

    # K-points: high-symmetry + a few interior points
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # X
        (0.5, 0.5),      # M
        (0.25, 0.0),
        (0.5, 0.25),
        (0.25, 0.25),
        (0.125, 0.0),
        (0.375, 0.125),
        (0.125, 0.125),
        (0.375, 0.375),
    ]

    n_bands = 5
    max_omega4 = 0.0
    min_omega5 = float('inf')

    for kx_u, ky_u in k_points:
        kx_vec = kx_u + Gx_arr
        ky_vec = ky_u + Gy_arr
        k_sq = kx_vec ** 2 + ky_vec ** 2

        # Build H matrix
        H = np.zeros((Npw, Npw), dtype=np.float64)
        for i in range(Npw):
            for j in range(Npw):
                dx = int(round(Gx_arr[i] - Gx_arr[j]))
                dy = int(round(Gy_arr[i] - Gy_arr[j]))
                coeff = coeff_cache.get((dx, dy), 0.0)
                H[i, j] = coeff * k_sq[j]

        # Solve eigenvalue problem
        eigvals = np.linalg.eigvals(H)
        eigvals_real = np.real(eigvals)
        eigvals_pos = eigvals_real[eigvals_real > 1e-10]
        eigvals_pos.sort()

        if len(eigvals_pos) >= n_bands:
            omegas = np.sqrt(eigvals_pos[:n_bands])
            if len(omegas) >= 5:
                max_omega4 = max(max_omega4, float(omegas[3]))
                min_omega5 = min(min_omega5, float(omegas[4]))

    if min_omega5 == float('inf') or max_omega4 < 1e-10:
        return -1.0

    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0

    if midgap < 1e-10:
        return -1.0

    return 2.0 * gap / midgap

def _make_harmonic_cell(params, size=32):
    """
    Create a C4v-symmetric cell from harmonic potential parameters.
    params: list of (amplitude, phase_x, phase_y, order) for each harmonic component
    """
    cell = np.zeros((size, size), dtype=np.float64)
    cx, cy = size / 2.0 - 0.5, size / 2.0 - 0.5

    for i in range(size):
        for j in range(size):
            # Normalize coordinates to [-1, 1]
            x = (j - cx) / (size / 2.0)
            y = (i - cy) / (size / 2.0)
            r = np.sqrt(x * x + y * y)
            theta = np.arctan2(y, x)

            val = 0.0
            for (amp, kx, ky, order) in params:
                # Use cos(kx*x + ky*y + phase) type terms with C4v symmetry
                # C4v symmetric terms: cos(4*theta), cos(8*theta), etc.
                # Or use polynomial basis
                term = np.cos(kx * x + ky * y)
                val += amp * term

            cell[i, j] = val

    # Threshold at zero
    cell = (cell > 0).astype(np.int8)
    return _enforce_c4v(cell.tolist())

def _make_rod_cell(rod_radius, centers, size=32):
    """Create a cell with dielectric rods at given centers."""
    cell = np.zeros((size, size), dtype=np.int8)
    for cy, cx in centers:
        for i in range(size):
            for j in range(size):
                if (i - cy) ** 2 + (j - cx) ** 2 < rod_radius ** 2:
                    cell[i, j] = 1
    return _enforce_c4v(cell.tolist())

def _make_superellipse_cell(exponent, width, size=32):
    """Create a superellipse-shaped dielectric region."""
    cell = np.zeros((size, size), dtype=np.int8)
    cx, cy = size / 2.0 - 0.5, size / 2.0 - 0.5
    half = size / 2.0

    for i in range(size):
        for j in range(size):
            x = (j - cx) / half
            y = (i - cy) / half
            # Superellipse: |x/a|^n + |y/b|^n <= 1
            val = abs(x / width) ** exponent + abs(y / width) ** exponent
            if val <= 1.0:
                cell[i, j] = 1
    return _enforce_c4v(cell.tolist())

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell that maximizes the TM band gap
    between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 220  # Safety margin

    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    # Generate diverse initial configurations
    initial_configs = []

    # Rod configurations with various radii and positions
    for radius in [2, 3, 4, 5]:
        for offset in [6, 8, 10]:
            centers = [(offset, offset), (offset, size - 1 - offset),
                       (size - 1 - offset, offset), (size - 1 - offset, size - 1 - offset)]
            initial_configs.append(_make_rod_cell(radius, centers, size))

    # Superellipse configurations
    for exponent in [2, 3, 4, 6]:
        for width in [0.5, 0.6, 0.7, 0.8]:
            initial_configs.append(_make_superellipse_cell(exponent, width, size))

    # Harmonic potential configurations
    for n_harmonics in [1, 2, 3]:
        for _ in range(3):
            params = []
            for _ in range(n_harmonics):
                amp = random.uniform(0.5, 2.0)
                kx = random.choice([4, 8, 12])
                ky = random.choice([0, 4, 8])
                order = random.choice([1, 2, 3])
                params.append((amp, kx, ky, order))
            initial_configs.append(_make_harmonic_cell(params, size))

    # Evaluate all initial configurations
    best_cell = None
    best_score = -1.0

    for cfg in initial_configs:
        if time.time() > deadline - 60:
            break
        score = _compute_gap_fast(cfg, problem)
        if score > best_score:
            best_score = score
            best_cell = cfg

    if best_cell is None:
        # Default fallback
        best_cell = _make_rod_cell(3, [(8, 8), (8, 24), (24, 8), (24, 24)], size)
        best_score = _compute_gap_fast(best_cell, problem)

    # Local search with pixel flips
    current_cell = np.array(best_cell, dtype=np.int8)
    current_score = best_score

    # Use simulated annealing
    temperature = 0.05
    cooling_rate = 0.995
    max_iter = 300

    for it in range(max_iter):
        if time.time() > deadline - 30:
            break

        # Pick a random pixel in the fundamental domain
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

        new_score = _compute_gap_fast(new_cell.tolist(), problem)

        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / temperature):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = new_cell.tolist()

        temperature *= cooling_rate

    # Final symmetry check
    best_cell = _enforce_c4v(best_cell)

    return best_cell
# EVOLVE-BLOCK-END
