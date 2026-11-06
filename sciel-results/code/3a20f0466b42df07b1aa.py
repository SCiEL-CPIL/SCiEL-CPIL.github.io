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
    DEADLINE = 215.0  # Safety margin before 230s kill
    
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

    def generate_harmonic_shape(R0, amps, phases):
        """
        Generate a cell with a harmonic boundary.
        r(theta) = R0 * (1 + sum(amp_k * cos(k*theta + phase_k)))
        amps and phases are lists corresponding to k = 2, 4, 6, 8, 10
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        r_func = R0 * np.ones_like(R)
        ks = [2, 4, 6, 8, 10]
        for i, k in enumerate(ks):
            if i < len(amps):
                r_func += amps[i] * np.cos(k * Theta + phases[i])
                
        # Fill where R <= r_func
        # Note: r_func can be negative, which results in empty cell or full cell depending on implementation
        # We want dielectric (1) where R <= r_func
        cell[R <= r_func] = 1
        
        return enforce_symmetry(cell)

    def generate_superellipses(n_exp, a_size, offset):
        """
        Generate a cell with 4 superellipses at (+/-offset, +/-offset).
        Equation: |x/a|^n + |y/a|^n <= 1
        """
        cell = np.zeros((size, size), dtype=int)
        # Coordinates relative to center, scaled to [-0.5, 0.5]
        # Pixel centers are at (i+0.5)/size - 0.5
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            # Superellipse condition
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
            
        return enforce_symmetry(cell)

    # Phase 1: Global Search
    # We will search in the Harmonic family primarily, as it is more flexible for a single central object.
    # We also include a few Superellipse points as a baseline.
    
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    best_family = None
    
    # 1a. Coarse Search in Harmonic Family
    # R0 in [0.2, 0.45]
    # A2 in [-0.1, 0.1]
    # A4 in [-0.2, 0.2]
    # A6 in [-0.1, 0.1]
    # A8 in [-0.1, 0.1]
    # Phases fixed to 0 for simplicity in coarse grid (symmetry handles the rest)
    
    R0_vals = [0.25, 0.30, 0.35, 0.40]
    A2_vals = [-0.05, 0.0, 0.05]
    A4_vals = [-0.10, -0.05, 0.0, 0.05, 0.10]
    A6_vals = [0.0] # Keep low order for coarse
    A8_vals = [0.0]
    
    for R0 in R0_vals:
        for A2 in A2_vals:
            for A4 in A4_vals:
                if not time_ok():
                    break
                amps = [A2, A4, 0.0, 0.0, 0.0]
                phases = [0.0, 0.0, 0.0, 0.0, 0.0]
                cell = generate_harmonic_shape(R0, amps, phases)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_params = (R0, amps, phases)
                    best_cell = cell.copy()
                    best_family = 'harmonic'
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    
    # 1b. Check a few Superellipse baselines to ensure we don't miss the 4-blob topology
    # The 4-blob topology is distinct from the central disk topology.
    # We should search both.
    
    n_values = [2.0, 3.0, 4.0]
    a_values = [0.10, 0.12, 0.15]
    off_values = [0.22, 0.25, 0.28]
    
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

    if best_cell is None:
        # Fallback
        best_cell = generate_harmonic_shape(0.35, [0,0,0,0,0], [0,0,0,0,0])
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))
        best_params = (0.35, [0,0,0,0,0], [0,0,0,0,0])
        best_family = 'harmonic'

    # Phase 2: Local Optimization
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 130.0  # Leave ~80s for pixel refinement and safety
    
    if best_family == 'harmonic':
        current_R0, current_amps, current_phases = best_params
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            # Perturb parameters
            new_R0 = current_R0 * (1.0 + np.random.normal(0, 0.05))
            new_R0 = np.clip(new_R0, 0.15, 0.48)
            
            new_amps = list(current_amps)
            new_phases = list(current_phases)
            
            # Perturb amplitudes
            for i in range(len(new_amps)):
                new_amps[i] += np.random.normal(0, 0.05)
                # Clip amplitudes to avoid huge negative radii or extreme shapes
                new_amps[i] = np.clip(new_amps[i], -0.3, 0.3)
            
            # Perturb phases slightly (though symmetry might wash this out, it helps break degeneracies in intermediate steps)
            for i in range(len(new_phases)):
                new_phases[i] += np.random.normal(0, 0.2)
            
            new_cell = generate_harmonic_shape(new_R0, new_amps, new_phases)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_R0 = new_R0
                current_amps = new_amps
                current_phases = new_phases
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    elif best_family == 'superellipse':
        current_n, current_a, current_off = best_params
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
            new_n = current_n * (1.0 + np.random.normal(0, 0.1))
            new_a = current_a * (1.0 + np.random.normal(0, 0.1))
            new_off = current_off + np.random.normal(0, 0.02)
            
            new_n = np.clip(new_n, 1.0, 10.0)
            new_a = np.clip(new_a, 0.05, 0.20)
            new_off = np.clip(new_off, 0.15, 0.35)
            
            new_cell = generate_superellipses(new_n, new_a, new_off)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-6:
                current_cell = new_cell
                current_ratio = new_ratio
                current_n, current_a, current_off = new_n, new_a, new_off
                no_improve_count = 0
            else:
                no_improve_count += 1

    # Phase 3: Pixel Refinement
    # This phase is critical to escape the parametric constraint and optimize the binary mask directly.
    # We perform stochastic local search on the pixels.
    
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 40:
        # Identify independent pixels under symmetry
        # For a 32x32 grid with D4 symmetry, we can pick pixels in the top-left 8x8 quadrant roughly,
        # but it's safer to pick all pixels and enforce symmetry after each flip.
        # To speed up, we only flip pixels that are likely to be on the boundary?
        # Or just random pixels.
        
        # Let's do a few passes of random pixel flips
        refine_start = time.time()
        refine_limit = 35.0
        
        no_improve_refine = 0
        max_no_improve_refine = 100
        
        # Get list of all pixel indices
        all_pixels = [(i, j) for i in range(size) for j in range(size)]
        
        while time.time() - refine_start < refine_limit and no_improve_refine < max_no_improve_refine and time_ok():
            # Pick a random pixel
            i, j = np.random.choice(size, 2)
            
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            # Check if the cell actually changed (symmetry might have reverted it or merged it)
            if np.array_equal(new_cell, current_cell):
                no_improve_refine += 1
                continue
                
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        # If we somehow degraded significantly, revert to best parametric cell
        # But usually pixel refinement only improves or stays same.
        pass # We just return current_cell which is the result of the process
        
    return current_cell.tolist()
# EVOLVE-BLOCK-END
