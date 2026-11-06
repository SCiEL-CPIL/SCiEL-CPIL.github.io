"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    # --- Fast TM plane-wave solver (17x17 = 289 plane waves) ---

    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # K-points in irreducible BZ (30 points)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)

    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = int(Gx[i]) - int(Gx[j])
            dGy = int(Gy[i]) - int(Gy[j])
            idx_x = dGx + half
            idx_y = dGy + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def compute_band_ratio_fast(coeffs):
        """Fast approximate evaluation using subset of k-points."""
        # Use more k-points for better guidance
        k_subset = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29]  # 11 representative k-points
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in k_subset:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def evaluate_cell(cell):
        """Full evaluation of a cell."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio_full(coeffs)

    def evaluate_cell_fast(cell):
        """Fast evaluation for search guidance."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio_fast(coeffs)

    # --- Symmetry enforcement: full D4 group ---

    def enforce_symmetry(cell):
        """Enforce full square (D4) symmetry by averaging over all 8 elements."""
        cell = np.array(cell, dtype=int)
        transforms = [
            cell,
            cell[::-1, :].copy(),      # vertical flip
            cell[:, ::-1].copy(),       # horizontal flip
            cell[::-1, ::-1].copy(),    # 180 rotation
            cell.T.copy(),              # transpose (main diagonal)
            cell.T[::-1, :].copy(),     # transpose + vflip = 90 CW
            cell.T[:, ::-1].copy(),     # transpose + hflip = 90 CCW
            cell[::-1, ::-1].T.copy(),  # anti-diagonal
        ]
        result = np.sum(transforms, axis=0)
        result = (result >= 4).astype(int)
        return result

    def enforce_symmetry_relaxed(cell):
        """Enforce symmetry with relaxed threshold to allow more dielectric."""
        cell = np.array(cell, dtype=int)
        transforms = [
            cell,
            cell[::-1, :].copy(),
            cell[:, ::-1].copy(),
            cell[::-1, ::-1].copy(),
            cell.T.copy(),
            cell.T[::-1, :].copy(),
            cell.T[:, ::-1].copy(),
            cell[::-1, ::-1].T.copy(),
        ]
        result = np.sum(transforms, axis=0)
        result = (result >= 3).astype(int)  # relaxed: >=3 of 8
        return result

    # --- Candidate generation ---

    def generate_rod(radius):
        """Single centered rod."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rods(radius, offset):
        """Four rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_ring(inner_r, outer_r):
        """Dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_cross(arm_width, arm_length):
        """Cross/plus shape."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                    cell[i, j] = 1
                if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rings(inner_r, outer_r, offset):
        """Four small rings at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        r2 = dx * dx + dy * dy
                        if inner_r * inner_r < r2 < outer_r * outer_r:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_square_ring(inner_r, outer_r):
        """Square dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r_max = max(abs(x), abs(y))
                if inner_r < r_max < outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_checkerboard(block_size, dielectric_frac):
        """Checkerboard pattern with block_size x block_size blocks."""
        cell = np.zeros((size, size), dtype=int)
        n_blocks = size // block_size
        for bi in range(n_blocks):
            for bj in range(n_blocks):
                if (bi + bj) % 2 == 0:
                    for i in range(bi * block_size, (bi + 1) * block_size):
                        for j in range(bj * block_size, (bj + 1) * block_size):
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_superlattice_disk(offsets, radii):
        """4x4 superlattice of 8x8 sub-cells, each a circular dielectric disk."""
        cell = np.zeros((size, size), dtype=int)
        sub_size = size // 4  # 8
        for block_i in range(4):
            for block_j in range(4):
                cx = (block_i + 0.5) * sub_size / size - 0.5 + offsets[block_i, block_j]
                cy = (block_j + 0.5) * sub_size / size - 0.5 + offsets[block_j, block_i]
                r = radii[block_i, block_j]
                for i in range(size):
                    for j in range(size):
                        x = (i + 0.5) / size - 0.5
                        y = (j + 0.5) / size - 0.5
                        if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_moire(f1, f2, phase1, phase2, threshold):
        """Moiré pattern from two counter-rotating square waves."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                v1 = np.sin(2 * np.pi * f1 * (x * np.cos(phase1) + y * np.sin(phase1)))
                v2 = np.sin(2 * np.pi * f2 * (x * np.cos(phase2) + y * np.sin(phase2)))
                if v1 * v2 > threshold:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    # --- Generate diverse candidates ---

    candidates = []

    # Single rods
    for r in [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        candidates.append(generate_rod(r))

    # Four-rod patterns
    for r in [0.05, 0.07, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(generate_four_rods(r, off))

    # Rings
    for inner_r in [0.10, 0.15, 0.20, 0.25]:
        for outer_r in [0.25, 0.30, 0.35, 0.40]:
            if outer_r > inner_r + 0.05:
                candidates.append(generate_ring(inner_r, outer_r))

    # Square rings
    for inner_r in [0.10, 0.15, 0.20]:
        for outer_r in [0.25, 0.30, 0.35]:
            if outer_r > inner_r + 0.05:
                candidates.append(generate_square_ring(inner_r, outer_r))

    # Crosses
    for w in [0.08, 0.10, 0.12, 0.15, 0.20]:
        for l in [0.25, 0.30, 0.35, 0.40, 0.45]:
            candidates.append(generate_cross(w, l))

    # Four rings
    for inner_r in [0.03, 0.05, 0.06, 0.08]:
        for outer_r in [0.08, 0.10, 0.12, 0.14]:
            if outer_r > inner_r + 0.02:
                for off in [0.18, 0.22, 0.25, 0.28, 0.30]:
                    candidates.append(generate_four_rings(inner_r, outer_r, off))

    # Checkerboards
    for bs in [4, 8]:
        candidates.append(generate_checkerboard(bs, 0.5))

    # Moiré patterns
    for f1 in [2, 3, 4]:
        for f2 in [3, 4, 5]:
            for p1 in [0, np.pi/4, np.pi/2]:
                for p2 in [np.pi/4, np.pi/2, 3*np.pi/4]:
                    for thresh in [-0.3, -0.1, 0.1, 0.3]:
                        candidates.append(generate_moire(f1, f2, p1, p2, thresh))

    # Superlattice patterns
    for mod_pattern in range(6):
        offsets = np.zeros((4, 4))
        radii = np.zeros((4, 4))
        for bi in range(4):
            for bj in range(4):
                if mod_pattern == 0:
                    radii[bi, bj] = 0.12 if (bi + bj) % 2 == 0 else 0.08
                elif mod_pattern == 1:
                    d = max(abs(bi - 1.5), abs(bj - 1.5))
                    radii[bi, bj] = 0.14 - 0.03 * d
                elif mod_pattern == 2:
                    if bi == 1 or bi == 2 or bj == 1 or bj == 2:
                        radii[bi, bj] = 0.13
                    else:
                        radii[bi, bj] = 0.07
                elif mod_pattern == 3:
                    radii[bi, bj] = 0.15 if (bi + bj) % 2 == 0 else 0.06
                elif mod_pattern == 4:
                    d = abs(bi - bj)
                    radii[bi, bj] = 0.14 - 0.04 * d
                else:
                    d = abs(bi + bj - 3)
                    radii[bi, bj] = 0.14 - 0.03 * d
                offsets[bi, bj] = 0.0
        candidates.append(generate_superlattice_disk(offsets, radii))

    # Random symmetric cells
    for _ in range(20):
        cell = np.zeros((size, size), dtype=int)
        # Fill independent quadrant randomly
        for i in range(size // 2):
            for j in range(size // 2):
                if random.random() < 0.4:
                    cell[i, j] = 1
        cell = enforce_symmetry(cell)
        candidates.append(cell)

    # --- Evaluate all candidates with fast evaluator ---

    best_ratio = -np.inf
    best_cell = None

    for cell in candidates:
        if time.time() - t_start > DEADLINE - 60:
            break
        ratio = evaluate_cell_fast(cell)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()

    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
        best_ratio = evaluate_cell_fast(best_cell)

    # --- Local search: flip individual pixels in the independent quadrant ---

    current_cell = best_cell.copy()
    current_ratio = best_ratio

    # Independent pixels: top-left quadrant (due to full square symmetry)
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]

    no_improve = 0
    max_no_improve = 40
    threshold = 1e-6

    search_time_limit = DEADLINE - 30  # Leave 30s for final eval and safety

    # Multi-pass local search with random restarts
    for pass_num in range(5):
        if time.time() - t_start > search_time_limit:
            break

        no_improve = 0
        while time.time() - t_start < search_time_limit and no_improve < max_no_improve:
            improved = False
            np.random.shuffle(indep_pixels)

            for (i, j) in indep_pixels:
                if time.time() - t_start > search_time_limit:
                    break

                old_val = current_cell[i, j]
                current_cell[i, j] = 1 - old_val
                current_cell = enforce_symmetry(current_cell)
                new_ratio = evaluate_cell_fast(current_cell)

                if new_ratio > current_ratio + threshold:
                    current_ratio = new_ratio
                    improved = True
                    no_improve = 0
                else:
                    current_cell[i, j] = old_val
                    current_cell = enforce_symmetry(current_cell)
                    no_improve += 1

                if time.time() - t_start > search_time_limit:
                    break

            if not improved:
                no_improve += 1
                # If stuck, try a random perturbation
                if no_improve > 10 and pass_num < 4:
                    for _ in range(5):
                        ri, rj = random.randint(0, size // 2 - 1), random.randint(0, size // 2 - 1)
                        current_cell[ri, rj] = 1 - current_cell[ri, rj]
                    current_cell = enforce_symmetry(current_cell)
                    current_ratio = evaluate_cell_fast(current_cell)
                    no_improve = 0

    # --- Final evaluation with full k-points ---

    if time.time() - t_start < DEADLINE - 5:
        coeffs = get_fourier_coeffs(current_cell)
        final_ratio = compute_band_ratio_full(coeffs)

    return current_cell.tolist()
# EVOLVE-BLOCK-END
