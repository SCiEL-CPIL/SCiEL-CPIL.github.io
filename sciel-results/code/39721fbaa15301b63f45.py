import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using topological optimization driven by
    local field intensities of the 4th and 5th TM bands.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Use 9x9 = 81 plane waves for surrogate (balance accuracy vs speed)
    N_PW = 9
    
    # --- PWE Solver with eigenvectors ---
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        1 -> dielectric (eps_high), 0 -> air (eps_low).
        """
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs

    def get_G_vectors(n_pw):
        """
        Returns G vectors (as integer multiples of 2pi/a) for square truncation.
        """
        half = n_pw // 2
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        return G_list

    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g if g >= 0 else N + g

    def solve_band_structure(cell, kx, ky, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        """
        fft_coeffs = get_fourier_coeffs(cell)
        G_vectors = get_G_vectors(N_PW)
        n = len(G_vectors)
        
        # Precompute G differences
        Gx_arr = np.array([g[0] for g in G_vectors])
        Gy_arr = np.array([g[1] for g in G_vectors])
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                Gx_diff = Gx_arr[i] - Gx_arr[j]
                Gy_diff = Gy_arr[i] - Gy_arr[j]
                idx_x = fft_index(Gx_diff, size)
                idx_y = fft_index(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        Kx_arr = kx + Gx_arr
        Ky_arr = ky + Gy_arr
        K_sq = Kx_arr**2 + Ky_arr**2
        
        # A = diag(K_sq) - M
        A = np.diag(K_sq) - M
        
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            idx_sort = np.argsort(eigvals)
            eigvals = eigvals[idx_sort]
            if return_eigvecs:
                eigvecs = eigvecs[:, idx_sort]
                return eigvals, eigvecs
            return eigvals
        except np.linalg.LinAlgError:
            if return_eigvecs:
                return np.zeros(n), np.zeros((n, n), dtype=complex)
            return np.zeros(n)

    def compute_field_intensities(cell, k_points, band_idx):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        Returns a 2D array of shape (size, size).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        G_vectors = get_G_vectors(N_PW)
        n = len(G_vectors)
        
        # Precompute pixel positions
        px_range = np.arange(size)
        ry, rx = np.meshgrid(px_range / size, px_range / size, indexing='ij')
        
        # Precompute phases for all G vectors at all k points
        # phase[i, kx_idx, y, x] = 2*pi*(Kx_i * rx[y,x] + Ky_i * ry[y,x])
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            # Get eigenvector for the specified band
            vec = eigvecs[:, band_idx]
            
            # Compute field at each pixel
            field = np.zeros((size, size), dtype=complex)
            for i in range(n):
                gx_i, gy_i = G_vectors[i]
                Kx = kx + gx_i
                Ky = ky + gy_i
                phase = 2 * np.pi * (Kx * rx + Ky * ry)
                field += vec[i] * np.exp(1j * phase)
            
            total_intensity += np.abs(field)**2
            count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity

    def estimate_gap(cell, k_points):
        """
        Estimates the gap between band 4 and 5.
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky)
            if len(eigvals) < 5:
                continue
            w4 = np.sqrt(max(0, eigvals[3]))
            w5 = np.sqrt(max(0, eigvals[4]))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0
        gap = min_omega_5 - max_omega_4
        return gap

    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique mask.
        The cell has 4-fold rotational symmetry and mirror symmetry.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
                # 8 symmetric positions (may have duplicates on axes)
                coords = set()
                coords.add((i, j))
                coords.add((size-1-i, j))
                coords.add((i, size-1-j))
                coords.add((size-1-i, size-1-j))
                coords.add((j, i))
                coords.add((size-1-j, i))
                coords.add((j, size-1-i))
                coords.add((size-1-j, size-1-i))
                for r, c in coords:
                    cell[r][c] = val
        return cell

    def get_symmetric_coords(i, j):
        """Get all symmetric coordinates for pixel (i,j) in 16x16 quadrant."""
        coords = set()
        coords.add((i, j))
        coords.add((size-1-i, j))
        coords.add((i, size-1-j))
        coords.add((size-1-i, size-1-j))
        coords.add((j, i))
        coords.add((size-1-j, i))
        coords.add((j, size-1-i))
        coords.add((size-1-j, size-1-i))
        return list(coords)

    # --- Optimization ---
    
    # Define k-points for the surrogate - comprehensive coverage of irreducible BZ
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.1),
        (0.4, 0.1),
        (0.1, 0.4),
        (0.3, 0.2),
        (0.2, 0.3),
        (0.15, 0.0),
        (0.35, 0.0),
        (0.0, 0.15),
        (0.0, 0.35),
        (0.45, 0.45),
        (0.55, 0.45),
        (0.45, 0.55),
        (0.55, 0.55),
        (0.2, 0.4),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.1, 0.3),
        (0.3, 0.1),
        (0.4, 0.4),
        (0.2, 0.2),
    ]
    
    # Initialize with a random binary mask
    np.random.seed(42)
    unique_mask = np.random.randint(0, 2, (16, 16)).astype(int)
    
    cell = fill_symmetric(unique_mask)
    current_gap = estimate_gap(cell, k_points)
    best_gap = current_gap
    best_cell = cell.copy()
    
    # Precompute symmetric coordinate lists for all unique pixels
    unique_pixels = [(i, j) for i in range(16) for j in range(16)]
    sym_coords = {p: get_symmetric_coords(p[0], p[1]) for p in unique_pixels}
    
    # Topological optimization loop
    no_improve_count = 0
    max_no_improve = 40
    batch_sizes = [1, 2, 3, 4, 5]
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        intensity_4 = compute_field_intensities(cell, k_points, 3)  # Band 4 (0-indexed: 3)
        intensity_5 = compute_field_intensities(cell, k_points, 4)  # Band 5 (0-indexed: 4)
        
        # Compute a "pressure" field: positive means add dielectric, negative means remove
        pressure = intensity_4 - intensity_5
        
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        else:
            pressure = np.zeros_like(pressure)
        
        # Compute average pressure for each unique pixel
        pixel_pressures = []
        for (i, j) in unique_pixels:
            coords = sym_coords[(i, j)]
            avg_pressure = np.mean([pressure[r, c] for r, c in coords])
            pixel_pressures.append(((i, j), avg_pressure))
        
        # Sort by absolute pressure (descending)
        pixel_pressures.sort(key=lambda x: abs(x[1]), reverse=True)
        
        improved = False
        
        # Try different batch sizes
        for batch_size in batch_sizes:
            if time.time() - start_time > time_limit:
                break
            
            # Select top batch_size pixels to modify
            selected = pixel_pressures[:batch_size]
            
            new_mask = unique_mask.copy()
            for (i, j), avg_p in selected:
                if avg_p > 0:
                    new_mask[i][j] = 1
                else:
                    new_mask[i][j] = 0
            
            new_cell = fill_symmetric(new_mask)
            new_gap = estimate_gap(new_cell, k_points)
            
            if new_gap > current_gap:
                current_gap = new_gap
                unique_mask = new_mask
                cell = new_cell
                if new_gap > best_gap:
                    best_gap = new_gap
                    best_cell = new_cell.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                improved = True
                break
            else:
                no_improve_count += 1
        
        if not improved:
            # Random perturbation to escape local minima
            n_rand = min(3, len(unique_pixels))
            for _ in range(n_rand):
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
            cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(cell, k_points)
            if new_gap > current_gap:
                current_gap = new_gap
                if new_gap > best_gap:
                    best_gap = new_gap
                    best_cell = new_cell.copy()
                    no_improve_count = 0
            else:
                # Revert to best
                for (i, j) in unique_pixels:
                    unique_mask[i][j] = best_cell[i][j]
                cell = fill_symmetric(unique_mask)
                current_gap = best_gap
                no_improve_count += 1
        
        # Early termination if stuck for too long
        if no_improve_count > max_no_improve:
            # More aggressive random restart
            for _ in range(10):
                ri = random.randint(0, 15)
                rj = random.randint(0, 15)
                unique_mask[ri][rj] = 1 - unique_mask[ri][rj]
            cell = fill_symmetric(unique_mask)
            new_gap = estimate_gap(cell, k_points)
            if new_gap > current_gap:
                current_gap = new_gap
                if new_gap > best_gap:
                    best_gap = new_gap
                    best_cell = new_cell.copy()
                    no_improve_count = 0
            else:
                # Revert to best
                for (i, j) in unique_pixels:
                    unique_mask[i][j] = best_cell[i][j]
                cell = fill_symmetric(unique_mask)
                current_gap = best_gap
                no_improve_count = 0
    
    # Final check: ensure best_cell is valid
    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()
