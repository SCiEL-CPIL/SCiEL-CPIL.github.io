"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Optimize 32x32 dielectric pattern for max TM gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 215.0  # Safety margin before 230s kill
    
    # Plane wave grid: 17x17 = 289 plane waves, G from -8 to 8
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # Precompute k-points (30 points in irreducible BZ)
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
    
    # Precompute Fourier coefficient indices: for each pair (i,j),
    # the coefficient is eps(Gx[i]-Gx[j], Gy[i]-Gy[j]) which corresponds
    # to pixel index ((dGx) mod size, (dGy) mod size) in the 2D FFT
    diff_fft_ix = np.zeros((n_pw_total, n_pw_total), dtype=int)
    diff_fft_iy = np.zeros((n_pw_total, n_pw_total), dtype=int)
    diff_valid = np.zeros((n_pw_total, n_pw_total), dtype=bool)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            # The Fourier coefficient at frequency (dGx, dGy) is at
            # pixel position (dGx mod size, dGy mod size) in the FFT
            diff_fft_ix[i, j] = int(dGx) % size
            diff_fft_iy[i, j] = int(dGy) % size
            diff_valid[i, j] = True  # All differences are valid for pixel-based coefficients
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        # Extract the needed coefficients
        coeffs = fft_2d[diff_fft_ix.ravel(), diff_fft_iy.ravel()].reshape(n_pw_total, n_pw_total)
        return coeffs
    
    def compute_band_ratio(coeffs, k_subset=None):
        """Compute gap-midgap ratio for bands 4 and 5."""
        if k_subset is None:
            k_indices = range(len(k_points))
        else:
            k_indices = k_subset
        
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
            k2 = k2_all[k_idx]
            # H[i,j] = (k+G_i)^2 * eps(G_i - G_j)
            H = k2[:, None] * coeffs
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0
    
    def make_symmetric(cell):
        """Enforce full square symmetry: cell == cell[::-1] rows/cols and cell == cell.T"""
        cell = np.array(cell, dtype=int)
        # Apply all 8 symmetries of the square group and take majority vote
        cell_rot180 = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        cell_rot90 = cell.T[::-1, :].copy()  # transpose then flip rows
        cell_rot270 = cell.T[:, ::-1].copy()  # transpose then flip cols
        cell_diag = cell.T.copy()
        cell_anti_diag = cell[::-1, ::-1].T.copy()
        
        result = (cell + cell_rot180 + cell_hflip + cell_vflip + 
                  cell_rot90 + cell_rot270 + cell_diag + cell_anti_diag)
        result = (result >= 4).astype(int)
        return result
    
    def initial_cell(radius, shape='circle'):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if shape == 'circle':
                    if x * x + y * y < radius * radius:
                        cell[i, j] = 1
                elif shape == 'square':
                    if abs(x) < radius and abs(y) < radius:
                        cell[i, j] = 1
                elif shape == 'ring':
                    r = np.sqrt(x * x + y * y)
                    if abs(r - radius) < 0.08:
                        cell[i, j] = 1
        return make_symmetric(cell)
    
    # Use fewer k-points for fast evaluation during search
    # Pick representative points from each edge
    fast_k_indices = [0, 5, 9, 10, 15, 20, 25, 29]
    
    def fast_compute_band_ratio(coeffs):
        return compute_band_ratio(coeffs, k_subset=fast_k_indices)
    
    # Evaluate initial guesses with fast evaluation
    best_ratio = -np.inf
    best_cell = None
    np.random.seed(42)
    
    initial_configs = []
    # Circles of various radii
    for radius in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        initial_configs.append(initial_cell(radius, 'circle'))
    # Rings
    for radius in [0.2, 0.25, 0.3, 0.35]:
        initial_configs.append(initial_cell(radius, 'ring'))
    # Squares
    for radius in [0.2, 0.25, 0.3]:
        initial_configs.append(initial_cell(radius, 'square'))
    
    for cell in initial_configs:
        if time.time() - start_time > time_limit * 0.15:
            break
        coeffs = get_fourier_coeffs(cell)
        ratio = fast_compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
    
    if best_cell is None:
        best_cell = initial_cell(0.3, 'circle')
        best_ratio = fast_compute_band_ratio(get_fourier_coeffs(best_cell))
    
    # Local search with fast evaluation
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    # Independent pixels: top-left quadrant (due to symmetry)
    indep_pixels = [(i, j) for i in range(size // 2) for j in range(size // 2)]
    
    no_improve = 0
    max_no_improve = 15
    threshold = 1e-6
    
    search_time_limit = time_limit * 0.75
    
    while time.time() - start_time < search_time_limit and no_improve < max_no_improve:
        improved = False
        np.random.shuffle(indep_pixels)
        
        for (i, j) in indep_pixels:
            if time.time() - start_time > search_time_limit:
                break
            
            old_val = current_cell[i, j]
            current_cell[i, j] = 1 - old_val
            current_cell = make_symmetric(current_cell)
            coeffs = get_fourier_coeffs(current_cell)
            new_ratio = fast_compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + threshold:
                current_ratio = new_ratio
                improved = True
                no_improve = 0
            else:
                current_cell[i, j] = old_val
                current_cell = make_symmetric(current_cell)
                no_improve += 1
            
            if time.time() - start_time > search_time_limit:
                break
        
        if not improved:
            no_improve += 1
    
    # Final evaluation with full k-points to get accurate score
    if time.time() - start_time < time_limit - 5:
        coeffs = get_fourier_coeffs(current_cell)
        final_ratio = compute_band_ratio(coeffs)
    
    return current_cell.tolist()
# EVOLVE-BLOCK-END
