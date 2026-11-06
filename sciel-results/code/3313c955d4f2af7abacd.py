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
        
    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
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
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        trans = cell.T.copy()
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    # Precompute coordinate grid for shape generation
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='xy')
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    def generate_superellipses(n_exp, a_size, offset):
        """Generate a cell with 4 superellipses at (+/-offset, +/-offset)."""
        cell = np.zeros((size, size), dtype=int)
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
        return enforce_symmetry(cell)
    
    def generate_rings(n_rings, r_inner_base, r_outer_base, thickness_base, ring_spacing, modulation=0.0):
        """
        Generate concentric dielectric rings.
        n_rings: number of rings (1-4)
        r_inner_base: base inner radius
        r_outer_base: base outer radius  
        thickness_base: base ring thickness
        ring_spacing: spacing between rings
        modulation: angular modulation amplitude for breaking perfect circular symmetry
        """
        cell = np.zeros((size, size), dtype=int)
        
        for ring in range(n_rings):
            r_in = r_inner_base + ring * ring_spacing
            r_out = r_in + thickness_base
            
            # Add angular modulation for more complex structure
            if modulation > 0:
                r_in_mod = r_in * (1.0 + modulation * np.cos(4 * Theta))
                r_out_mod = r_out * (1.0 + modulation * np.cos(4 * Theta + np.pi/4))
            else:
                r_in_mod = np.full_like(R, r_in)
                r_out_mod = np.full_like(R, r_out)
            
            mask = (R >= r_in_mod) & (R <= r_out_mod)
            cell[mask] = 1
        
        return enforce_symmetry(cell)
    
    def generate_hybrid(n_exp, a_size, offset, n_rings, r_base, thickness):
        """Hybrid: superellipses + central ring structure."""
        cell = np.zeros((size, size), dtype=int)
        
        # Superellipses
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
        
        # Central ring
        r_in = r_base
        r_out = r_base + thickness
        mask = (R >= r_in) & (R <= r_out)
        cell[mask] = 1
        
        return enforce_symmetry(cell)

    # Phase 1: Global Search over multiple design families
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    best_family = None
    
    # Family 1: Superellipses (proven approach)
    n_values = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    a_values = [0.08, 0.10, 0.12, 0.14]
    off_values = [0.20, 0.22, 0.25, 0.28]
    
    for n in n_values:
        for a in a_values:
            for off in off_values:
                if not time_ok():
                    break
                cell = generate_superellipses(n, a, off)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (n, a, off)
                    best_cell = cell.copy()
                    best_family = 'superellipse'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 2: Concentric rings (new structural family)
    n_ring_values = [1, 2, 3]
    r_inner_values = [0.10, 0.12, 0.15, 0.18]
    r_outer_values = [0.20, 0.25, 0.30]
    spacing_values = [0.08, 0.10, 0.12]
    mod_values = [0.0, 0.05, 0.10]
    
    for nr in n_ring_values:
        for ri in r_inner_values:
            for ro in r_outer_values:
                for sp in spacing_values:
                    for mod in mod_values:
                        if not time_ok():
                            break
                        # Check if rings fit within cell
                        max_r = ri + (nr - 1) * sp + (ro - ri)
                        if max_r > 0.45:
                            continue
                        cell = generate_rings(nr, ri, ro, ro - ri, sp, mod)
                        coeffs = get_fourier_coeffs(cell)
                        ratio = compute_band_ratio(coeffs)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_params = (nr, ri, ro, sp, mod)
                            best_cell = cell.copy()
                            best_family = 'rings'
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # Family 3: Hybrid (superellipses + central ring)
    n_hyb = [2.0, 3.0]
    a_hyb = [0.10, 0.12]
    off_hyb = [0.22, 0.25]
    r_hyb = [0.08, 0.10]
    th_hyb = [0.06, 0.08]
    
    for n in n_hyb:
        for a in a_hyb:
            for off in off_hyb:
                for r in r_hyb:
                    for th in th_hyb:
                        if not time_ok():
                            break
                        cell = generate_hybrid(n, a, off, 1, r, th)
                        coeffs = get_fourier_coeffs(cell)
                        ratio = compute_band_ratio(coeffs)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_params = (n, a, off, r, th)
                            best_cell = cell.copy()
                            best_family = 'hybrid'
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    if best_cell is None:
        best_cell = generate_superellipses(2.0, 0.12, 0.25)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (2.0, 0.12, 0.25)
        best_family = 'superellipse'

    # Phase 2: Local Optimization with multi-restart
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 140.0
    max_restarts = 5
    restarts_done = 0
    
    while time.time() - search_start < search_time_limit and time_ok():
        improved_in_restart = False
        no_improve_count = 0
        max_no_improve = 30
        
        # For restarts after the first, perturb the starting point
        if restarts_done > 0:
            # Random perturbation of the current best
            perturbed = current_cell.copy()
            n_perturb = np.random.randint(3, 8)
            for _ in range(n_perturb):
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                perturbed[i, j] = 1 - perturbed[i, j]
            current_cell = enforce_symmetry(perturbed)
            current_ratio = compute_band_ratio(get_fourier_coeffs(current_cell))
        
        # Hill climbing on pixel flips
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 20.0
        
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
                no_improve_count = 0
                improved_in_restart = True
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        restarts_done += 1
        if not improved_in_restart and restarts_done >= max_restarts:
            break
        if not time_ok():
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
