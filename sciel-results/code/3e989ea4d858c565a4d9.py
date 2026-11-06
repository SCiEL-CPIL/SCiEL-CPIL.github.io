"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 210.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - start_time < deadline
    
    # Plane wave setup: 17x17 = 289 plane waves
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
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
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
        """Fast approximate evaluation using subset of k-points for search guidance."""
        # Use fewer k-points for speed
        k_subset = [0, 4, 9, 14, 19, 24, 29]  # 7 representative k-points
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
    
    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
    
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
    
    # Generate diverse candidates
    candidates = []
    
    # Single rods
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        candidates.append(generate_rod(r))
    
    # Four-rod patterns (strong candidates for higher-order gaps)
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(generate_four_rods(r, off))
    
    # Ring patterns
    for ir, orr in [(0.10, 0.20), (0.12, 0.22), (0.15, 0.25), (0.18, 0.28)]:
        candidates.append(generate_ring(ir, orr))
    
    # Cross patterns
    for aw, al in [(0.10, 0.35), (0.15, 0.40), (0.20, 0.45), (0.25, 0.50)]:
        candidates.append(generate_cross(aw, al))
    
    # Four small rings
    for ir, orr in [(0.04, 0.10), (0.05, 0.12), (0.06, 0.14)]:
        for off in [0.22, 0.25, 0.28]:
            candidates.append(generate_four_rings(ir, orr, off))
    
    # Evaluate candidates with fast evaluator first, then full for top candidates
    best_ratio = -np.inf
    best_cell = None
    candidate_scores = []
    
    for idx, cell in enumerate(candidates):
        if not time_ok():
            break
        
        coeffs = get_fourier_coeffs(cell)
        fast_ratio = compute_band_ratio_fast(coeffs)
        candidate_scores.append((fast_ratio, idx, cell))
    
    # Sort by fast score and evaluate top candidates with full evaluator
    candidate_scores.sort(key=lambda x: -x[0])
    
    num_full_evals = min(20, len(candidate_scores))
    for fast_ratio, idx, cell in candidate_scores[:num_full_evals]:
        if not time_ok():
            break
        
        coeffs = get_fourier_coeffs(cell)
        full_ratio = compute_band_ratio_full(coeffs)
        
        if full_ratio > best_ratio:
            best_ratio = full_ratio
            best_cell = cell.copy()
    
    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
        best_ratio = compute_band_ratio_full(get_fourier_coeffs(best_cell))
    
    # Local search with fast evaluation for guidance, full for acceptance
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    
    no_improve = 0
    max_no_improve = 40
    threshold = 1e-6
    
    search_start = time.time()
    search_time_limit = 80.0  # reserve time for final full evaluation
    
    while time.time() - search_start < search_time_limit and no_improve < max_no_improve and time_ok():
        improved = False
        np.random.shuffle(indep_pixels)
        
        for (i, j) in indep_pixels:
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
            
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = enforce_symmetry(current_cell)
            coeffs = get_fourier_coeffs(current_cell)
            
            # Fast evaluation for screening
            fast_ratio = compute_band_ratio_fast(coeffs)
            
            # Only do full evaluation if fast score is promising
            if fast_ratio > current_ratio - 0.05:
                new_ratio = compute_band_ratio_full(coeffs)
            else:
                new_ratio = fast_ratio * 0.9  # penalize for not doing full eval
            
            if new_ratio > current_ratio + threshold:
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                current_cell[i, j] = old_val
                current_cell = enforce_symmetry(current_cell)
                no_improve += 1
            
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
        
        if not improved:
            no_improve += 1
    
    # Final full evaluation of best cell
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio_full(final_coeffs)
    
    # If final ratio is worse, try to recover
    if final_ratio < best_ratio:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
    
    return current_cell.tolist()
# EVOLVE-BLOCK-END
