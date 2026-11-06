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

    def generate_from_radii(radii_4x4):
        """
        Generate a 32x32 cell from a 4x4 array of 8x8 sub-cell disk radii.
        The cell is divided into 4x4 = 16 blocks of 8x8 pixels each.
        Each block has a circular dielectric disk at its center with the given radius.
        radii_4x4: 4x4 array of radii (in pixels, 0-4 range for 8x8 blocks)
        """
        cell = np.zeros((size, size), dtype=int)
        block_size = 8
        n_blocks = size // block_size  # 4
        
        for bi in range(n_blocks):
            for bj in range(n_blocks):
                r = radii_4x4[bi, bj]
                if r <= 0:
                    continue
                # Center of this block
                ci = bi * block_size + block_size // 2
                cj = bj * block_size + block_size // 2
                # Create disk in this block
                for i in range(bi * block_size, (bi + 1) * block_size):
                    for j in range(bj * block_size, (bj + 1) * block_size):
                        dx = i - ci
                        dy = j - cj
                        if dx * dx + dy * dy <= r * r:
                            cell[i, j] = 1
        
        return enforce_symmetry(cell)

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

    def generate_four_roads(radius, offset):
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

    # Phase 1: Evaluate diverse initial candidates
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    param_type = None  # 'radii', 'superellipse', 'four_rods', 'single_rod'
    
    # Evaluate four-rod patterns
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        for off in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
            if not time_ok():
                break
            cell = generate_four_roads(r, off)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio(coeffs)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cell = cell.copy()
                best_params = (r, off)
                param_type = 'four_rods'
        if not time_ok():
            break
    
    # Evaluate single rods
    for r in [0.18, 0.22, 0.25, 0.28, 0.30, 0.32]:
        if not time_ok():
            break
        cell = generate_single_rod(r)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
            best_params = (r,)
            param_type = 'single_rod'
    
    # Evaluate superellipses
    for n in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
        for a in [0.08, 0.10, 0.12, 0.14]:
            for off in [0.20, 0.22, 0.25, 0.28]:
                if not time_ok():
                    break
                cell = generate_superellipses(n, a, off)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = (n, a, off)
                    param_type = 'superellipse'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Evaluate 4x4 radii superlattice patterns
    # Try several structured patterns
    if time_ok():
        # Pattern 1: uniform
        for r in [2.0, 2.5, 3.0, 3.5]:
            if not time_ok():
                break
            radii = np.full((4, 4), r)
            cell = generate_from_radii(radii)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio(coeffs)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cell = cell.copy()
                best_params = radii.copy()
                param_type = 'radii'
        
        # Pattern 2: checkerboard
        for r1 in [1.5, 2.0, 2.5]:
            for r2 in [3.0, 3.5, 4.0]:
                if not time_ok():
                    break
                radii = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        if (i + j) % 2 == 0:
                            radii[i, j] = r1
                        else:
                            radii[i, j] = r2
                cell = generate_from_radii(radii)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = radii.copy()
                    param_type = 'radii'
        
        # Pattern 3: ring pattern (larger in center)
        for rc in [3.5, 4.0]:
            for re in [1.5, 2.0]:
                if not time_ok():
                    break
                radii = np.full((4, 4), re)
                radii[1:3, 1:3] = rc
                cell = generate_from_radii(radii)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = radii.copy()
                    param_type = 'radii'
        
        # Pattern 4: cross pattern
        for rc in [3.0, 3.5, 4.0]:
            for re in [1.0, 1.5, 2.0]:
                if not time_ok():
                    break
                radii = np.full((4, 4), re)
                radii[1:3, :] = rc
                radii[:, 1:3] = rc
                cell = generate_from_radii(radii)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                    best_params = radii.copy()
                    param_type = 'radii'
    
    if best_cell is None:
        best_cell = generate_four_roads(0.10, 0.25)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (0.10, 0.25)
        param_type = 'four_rods'

    # Phase 2: Local Optimization
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    current_params = best_params.copy() if isinstance(best_params, np.ndarray) else best_params
    current_type = param_type
    
    search_start = time.time()
    search_time_limit = 150.0
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        improved = False
        
        if current_type == 'radii':
            # Perturb radii
            new_radii = current_params.copy()
            for _ in range(3):  # Perturb 3 random entries
                ri = np.random.randint(0, 4)
                rj = np.random.randint(0, 4)
                new_radii[ri, rj] += np.random.normal(0, 0.5)
                new_radii[ri, rj] = np.clip(new_radii[ri, rj], 0.5, 4.0)
            
            new_cell = generate_from_radii(new_radii)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_params = new_radii
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        elif current_type == 'superellipse':
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
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        elif current_type == 'four_rods':
            r, off = current_params
            new_r = r * (1.0 + np.random.normal(0, 0.1))
            new_off = off + np.random.normal(0, 0.02)
            
            new_r = np.clip(new_r, 0.04, 0.20)
            new_off = np.clip(new_off, 0.15, 0.35)
            
            new_cell = generate_four_roads(new_r, new_off)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_params = (new_r, new_off)
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        elif current_type == 'single_rod':
            r = current_params[0]
            new_r = r * (1.0 + np.random.normal(0, 0.1))
            new_r = np.clip(new_r, 0.10, 0.40)
            
            new_cell = generate_single_rod(new_r)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_params = (new_r,)
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 10
    
    # Phase 3: Pixel Refinement
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 30:
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 25.0
        
        no_improve_refine = 0
        for (i, j) in indep_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
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
