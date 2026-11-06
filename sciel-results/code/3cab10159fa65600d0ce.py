"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill
    
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
    
    # K-points in irreducible BZ (30 points as specified)
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
        
    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
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

    def generate_superellipses(n_exp, a_size, offset):
        """
        Generate a cell with 4 superellipses at (+/-offset, +/-offset).
        Equation: |x/a|^n + |y/a|^n <= 1
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
            
        return enforce_symmetry(cell)

    def generate_fractal_disk(r0, a4, a8, phase4=0.0, phase8=0.0):
        """
        Generate a centrally located dielectric disk with fractal boundary.
        r(theta) = r0 * (1 + a4*cos(4*theta + phase4) + a8*cos(8*theta + phase8))
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        r_func = r0 * (1.0 + a4 * np.cos(4 * Theta + phase4) + a8 * np.cos(8 * Theta + phase8))
        
        cell[R <= r_func] = 1
        
        return enforce_symmetry(cell)

    # Phase 1: Global Search
    best_ratio = -np.inf
    best_cell = None
    best_type = None
    best_params = None
    
    # 1a. Search Central Fractal Disks (Primary focus for 4-5 TM gap)
    r0_values = [0.20, 0.25, 0.30, 0.35, 0.40]
    a4_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    a8_values = [0.0, 0.1, 0.2]
    
    for r0 in r0_values:
        for a4 in a4_values:
            for a8 in a8_values:
                if not time_ok(): break
                cell = generate_fractal_disk(r0, a4, a8)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = (r0, a4, a8)
                    best_type = 'disk'
            if not time_ok(): break
        if not time_ok(): break
    
    # 1b. Search Four Superellipses (Secondary)
    n_values = [2.0, 3.0, 4.0, 6.0]
    a_values = [0.08, 0.10, 0.12]
    off_values = [0.20, 0.25, 0.30]
    
    for n in n_values:
        for a in a_values:
            for off in off_values:
                if not time_ok(): break
                cell = generate_superellipses(n, a, off)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = (n, a, off)
                    best_type = 'rod'
            if not time_ok(): break
        if not time_ok(): break

    if best_cell is None:
        best_cell = generate_fractal_disk(0.3, 0.2, 0.1)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (0.3, 0.2, 0.1)
        best_type = 'disk'

    # Phase 2: Local Optimization
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    current_params = best_params
    current_type = best_type
    
    search_start = time.time()
    search_time_limit = 140.0
    
    no_improve_count = 0
    max_no_improve = 60
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        improved = False
        
        if current_type == 'disk':
            r0, a4, a8 = current_params
            new_r0 = r0 * (1.0 + np.random.normal(0, 0.05))
            new_a4 = a4 * (1.0 + np.random.normal(0, 0.15)) + np.random.normal(0, 0.02)
            new_a8 = a8 * (1.0 + np.random.normal(0, 0.15)) + np.random.normal(0, 0.02)
            
            new_r0 = np.clip(new_r0, 0.15, 0.45)
            new_a4 = np.clip(new_a4, -0.5, 0.5)
            new_a8 = np.clip(new_a8, -0.5, 0.5)
            
            new_cell = generate_fractal_disk(new_r0, new_a4, new_a8)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_params = (new_r0, new_a4, new_a8)
                improved = True
        else:
            n, a, off = current_params
            new_n = n * (1.0 + np.random.normal(0, 0.1))
            new_a = a * (1.0 + np.random.normal(0, 0.1))
            new_off = off + np.random.normal(0, 0.02)
            
            new_n = np.clip(new_n, 1.0, 20.0)
            new_a = np.clip(new_a, 0.05, 0.20)
            new_off = np.clip(new_off, 0.15, 0.35)
            
            new_cell = generate_superellipses(new_n, new_a, new_off)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_params = (new_n, new_a, new_off)
                improved = True
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck for a while, try switching topology
            if no_improve_count % 20 == 0:
                if current_type == 'disk':
                    # Try a random rod
                    n = np.random.uniform(2, 8)
                    a = np.random.uniform(0.08, 0.15)
                    off = np.random.uniform(0.2, 0.3)
                    test_cell = generate_superellipses(n, a, off)
                    test_ratio = compute_band_ratio(get_fourier_coeffs(test_cell))
                    if test_ratio > current_ratio:
                        current_cell = test_cell
                        current_ratio = test_ratio
                        current_params = (n, a, off)
                        current_type = 'rod'
                        no_improve_count = 0
                else:
                    # Try a random disk
                    r0 = np.random.uniform(0.2, 0.4)
                    a4 = np.random.uniform(0.0, 0.4)
                    a8 = np.random.uniform(0.0, 0.3)
                    test_cell = generate_fractal_disk(r0, a4, a8)
                    test_ratio = compute_band_ratio(get_fourier_coeffs(test_cell))
                    if test_ratio > current_ratio:
                        current_cell = test_cell
                        current_ratio = test_ratio
                        current_params = (r0, a4, a8)
                        current_type = 'disk'
                        no_improve_count = 0

    # Phase 3: Pixel Refinement
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 30:
        # Get fundamental domain pixels (i < j, i < 16, j < 16, and i < size - j - 1 roughly)
        # Simpler: just iterate over all pixels, flip, enforce symmetry, check.
        # To be efficient, we only flip pixels in the first octant (i <= j, i < 16, j < 16, i + j < 32)
        # Actually, just shuffling all and trying is safer if we have time.
        # Let's try a limited set of pixels in the fundamental domain.
        fund_pixels = []
        for i in range(16):
            for j in range(16):
                # Check if (i,j) is in the fundamental domain of the square symmetry group
                # Symmetry group acts on (x,y). 
                # We want to pick one representative from each orbit.
                # Orbit size is usually 8.
                # Simple check: i < j and i < 16 and j < 16 and i+j < 32 is not quite right for full group.
                # Let's just pick i in [0, 15], j in [i, 15] and ensure i < 16, j < 16.
                # This covers the upper triangle of the top-left quadrant.
                if j >= i:
                    fund_pixels.append((i, j))
        
        np.random.shuffle(fund_pixels)
        
        refine_start = time.time()
        refine_limit = 25.0
        
        no_improve_refine = 0
        for (i, j) in fund_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
                
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            # Check if the flip actually changed the cell (symmetry enforcement might revert it if it was already symmetric in a way that forced it)
            if np.array_equal(new_cell, current_cell):
                continue
                
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1
                if no_improve_refine > 50:
                    break

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
        
    return current_cell.tolist()
# EVOLVE-BLOCK-END
