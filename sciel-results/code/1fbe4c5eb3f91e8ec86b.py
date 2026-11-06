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
    deadline = 220.0  # Safety margin before 230s kill
    
    def time_ok():
        return time.time() - start_time < deadline

    # Plane wave setup
    # Search uses 13x13 (169 PWs), Final uses 17x17 (289 PWs)
    n_pw_search = 13
    n_pw_final = 17
    
    def get_pw_grid(n_pw):
        half = n_pw // 2
        G_vals = np.arange(-half, half + 1)
        Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
        Gx = Gx_grid.ravel()
        Gy = Gy_grid.ravel()
        return Gx, Gy

    Gx_s, Gy_s = get_pw_grid(n_pw_search)
    Gx_f, Gy_f = get_pw_grid(n_pw_final)
    n_pw_total_s = n_pw_search * n_pw_search
    n_pw_total_f = n_pw_final * n_pw_final
    
    # K-points in irreducible BZ
    # Search: 10 points, Final: 30 points
    pi = np.pi
    def get_k_points(n_pts_per_edge):
        k_points = []
        for i in range(n_pts_per_edge):
            t = i / (n_pts_per_edge - 1) if n_pts_per_edge > 1 else 0
            k_points.append((t * pi, 0.0))
        for i in range(n_pts_per_edge):
            t = i / (n_pts_per_edge - 1) if n_pts_per_edge > 1 else 0
            k_points.append((pi, t * pi))
        for i in range(n_pts_per_edge):
            t = i / (n_pts_per_edge - 1) if n_pts_per_edge > 1 else 0
            k_points.append((pi * (1 - t), pi * (1 - t)))
        return np.array(k_points)
    
    k_points_s = get_k_points(4) # 12 points
    k_points_f = get_k_points(10) # 30 points
    
    def precompute_k2(k_points, Gx, Gy):
        k2_all = np.zeros((len(k_points), len(Gx)))
        for i, (kx, ky) in enumerate(k_points):
            k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        return k2_all

    k2_s = precompute_k2(k_points_s, Gx_s, Gy_s)
    k2_f = precompute_k2(k_points_f, Gx_f, Gy_f)

    def precompute_diff_idx(Gx, Gy, n_pw):
        diff_idx = np.zeros((len(Gx), len(Gx)), dtype=int)
        half = n_pw // 2
        for i in range(len(Gx)):
            for j in range(len(Gx)):
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
        return diff_idx, valid_rows, valid_cols

    diff_idx_s, valid_rows_s, valid_cols_s = precompute_diff_idx(Gx_s, Gy_s, n_pw_search)
    diff_idx_f, valid_rows_f, valid_cols_f = precompute_diff_idx(Gx_f, Gy_f, n_pw_final)

    def get_fourier_coeffs(cell, Gx, Gy, n_pw_total):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        half = n_pw_total // (n_pw_total ** 0.5) # This is wrong, n_pw_total is N^2
        # Actually, we can just use the Gx, Gy arrays
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs, k2_all, diff_idx, valid_rows, valid_cols, n_pw_total):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k2_all)):
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

    def evaluate_cell(cell, fast=True):
        if fast:
            coeffs = get_fourier_coeffs(cell, Gx_s, Gy_s, n_pw_total_s)
            return compute_band_ratio(coeffs, k2_s, diff_idx_s, valid_rows_s, valid_cols_s, n_pw_total_s)
        else:
            coeffs = get_fourier_coeffs(cell, Gx_f, Gy_f, n_pw_total_f)
            return compute_band_ratio(coeffs, k2_f, diff_idx_f, valid_rows_f, valid_cols_f, n_pw_total_f)

    def enforce_symmetry(cell):
        """Enforce C4v symmetry by constructing from top-left 16x16 block."""
        # The independent block is the top-left 16x16, but it must be symmetric itself
        # Actually, if we just take the top-left 16x16 and reflect, we get C2v.
        # To get C4v, we need cell[i,j] == cell[j,i].
        # So we should optimize the upper triangle of the 16x16 block.
        # But for simplicity, let's just enforce the full symmetry by averaging.
        cell = np.array(cell, dtype=float)
        # Symmetries:
        # 1. Identity
        # 2. Rotate 90
        # 3. Rotate 180
        # 4. Rotate 270
        # 5. Flip X
        # 6. Flip Y
        # 7. Flip X + Rotate 90 (Anti-diag)
        # 8. Flip Y + Rotate 90 (Diag)
        
        # Simpler: 
        # cell[i,j] = cell[31-i, j] (Vflip)
        # cell[i,j] = cell[i, 31-j] (Hflip)
        # cell[i,j] = cell[j, i] (Transpose)
        
        # We can just enforce these by iterating and averaging?
        # Or just construct from the top-left 16x16 block Q.
        # Q[i,j] must equal Q[j,i].
        # Let's assume the input cell is already roughly symmetric and we just snap it.
        
        # Better: Define the cell from the top-left 16x16 block.
        # But we are modifying pixels in the full 32x32 cell in the search.
        # So we need to enforce symmetry on the modified cell.
        
        # Let's use the averaging method for robustness.
        c = cell.copy()
        c += cell[::-1, :]
        c += cell[:, ::-1]
        c += cell[::-1, ::-1]
        c += cell.T
        c += cell.T[::-1, :]
        c += cell.T[:, ::-1]
        c += cell[::-1, ::-1].T
        c = (c / 8.0)
        return (c >= 0.5).astype(int)

    def generate_four_rods(radius, offset):
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

    # Generate diverse candidates
    candidates = []
    for r in [0.08, 0.10, 0.12, 0.15]:
        for off in [0.20, 0.25, 0.30]:
            candidates.append(generate_four_rods(r, off))
    
    # Evaluate candidates with fast evaluator
    best_ratio = -np.inf
    best_cell = None
    
    for cell in candidates:
        if not time_ok():
            break
        ratio = evaluate_cell(cell, fast=True)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
    
    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
        best_ratio = evaluate_cell(best_cell, fast=True)

    # Local search: flip individual pixels in the independent quadrant
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    # Independent pixels: top-left 16x16 quadrant, but only upper triangle to respect transpose symmetry?
    # Actually, if we flip a pixel in the top-left 16x16, and then enforce symmetry, 
    # the transpose symmetry might be broken if we don't handle it carefully.
    # But enforce_symmetry averages over all symmetries, so it will fix it.
    # So we can just iterate over all pixels in the top-left 16x16.
    
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    
    no_improve = 0
    max_no_improve = 10
    threshold = 1e-5
    
    search_time_limit = 200.0  # Leave 10s for final eval and safety
    
    # Greedy local search with restarts
    while time.time() - start_time < search_time_limit:
        improved = False
        np.random.shuffle(indep_pixels)
        
        for (i, j) in indep_pixels:
            if time.time() - start_time > search_time_limit:
                break
            
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = enforce_symmetry(current_cell)
            new_ratio = evaluate_cell(current_cell, fast=True)
            
            if new_ratio > current_ratio + threshold:
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                current_cell[i, j] = old_val
                current_cell = enforce_symmetry(current_cell)
                no_improve += 1
            
            if time.time() - start_time > search_time_limit:
                break
        
        if not improved:
            # Try a random kick to escape local minimum
            if no_improve > max_no_improve:
                # Flip 5 random pixels
                for _ in range(5):
                    i, j = np.random.randint(0, size//2), np.random.randint(0, size//2)
                    current_cell[i, j] = 1 - current_cell[i, j]
                current_cell = enforce_symmetry(current_cell)
                current_ratio = evaluate_cell(current_cell, fast=True)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve * 2:
                    break

    # Final evaluation with full k-points to get accurate score
    if time.time() - start_time < deadline - 5:
        coeffs = get_fourier_coeffs(current_cell, Gx_f, Gy_f, n_pw_total_f)
        final_ratio = compute_band_ratio(coeffs, k2_f, diff_idx_f, valid_rows_f, valid_cols_f, n_pw_total_f)
    
    return current_cell.tolist()
# EVOLVE-BLOCK-END
