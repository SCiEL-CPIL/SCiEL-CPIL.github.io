"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 210.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE
    
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
    n_valid = len(valid_rows)
    
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
    
    def compute_band_ratio(coeffs, k_indices=None):
        """Compute gap-midgap ratio."""
        if k_indices is None:
            k_indices = range(len(k_points))
        
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0
    
    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        # 4-fold rotation: 90, 180, 270 degrees
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        # Reflections
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        # Transpose
        trans = cell.T.copy()
        # All 8 symmetries
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result
    
    def generate_disk_with_harmonics(r_base, a4, a8):
        """Generate a centrally located disk with fractal boundary using angular harmonics."""
        cell = np.zeros((size, size), dtype=int)
        # Compute pixel centers
        x_coords = (np.arange(size) + 0.5) / size - 0.5
        y_coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Convert to polar
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Radial function with harmonics: r(theta) = r_base + a4*cos(4*theta) + a8*cos(8*theta)
        r_theta = r_base + a4 * np.cos(4 * Theta) + a8 * np.cos(8 * Theta)
        
        # Fill pixels where R <= r_theta
        mask = R <= r_theta
        cell[mask] = 1
        
        return enforce_symmetry(cell)
    
    def generate_four_rods(radius, offset):
        """Four dielectric rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy <= radius * radius:
                        cell[i, j] = 1
                        break
        return enforce_symmetry(cell)
    
    def generate_single_rod(radius):
        """Single centered dielectric rod."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y <= radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    # Phase 1: Evaluate a diverse set of initial candidates
    candidates = []
    
    # Disk with harmonics (primary new candidates)
    for r_base in [0.20, 0.25, 0.30, 0.35]:
        for a4 in [-0.08, -0.04, 0.0, 0.04, 0.08]:
            for a8 in [-0.06, -0.03, 0.0, 0.03, 0.06]:
                candidates.append(generate_disk_with_harmonics(r_base, a4, a8))
    
    # Four-rod patterns
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        for off in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(generate_four_rods(r, off))
    
    # Single rods
    for r in [0.18, 0.22, 0.25, 0.28, 0.30, 0.32]:
        candidates.append(generate_single_rod(r))
    
    # Evaluate with full k-point set
    best_ratio = -np.inf
    best_cell = None
    
    for idx, cell in enumerate(candidates):
        if not time_ok():
            break
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
    
    if best_cell is None:
        best_cell = generate_disk_with_harmonics(0.25, 0.0, 0.0)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
    
    # Phase 2: Local optimization - flip pixels in the independent quadrant
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    # Independent pixels: upper-left 16x16 quadrant (due to symmetry)
    indep_pixels = [(i, j) for i in range(16) for j in range(16)]
    
    no_improve_count = 0
    max_no_improve = 40
    search_start = time.time()
    search_time_limit = 150.0  # leave time for final eval
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        improved = False
        np.random.shuffle(indep_pixels)
        
        for (i, j) in indep_pixels:
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
            
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
        
        if not improved:
            no_improve_count += 1
    
    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    # If something went wrong, fall back to best found
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
    
    return current_cell.tolist()
