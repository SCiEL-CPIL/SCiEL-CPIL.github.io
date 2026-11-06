import numpy as np
import time
import random

def design(problem):
    """
    Topological optimization for 32x32 photonic crystal unit cell.
    Maximizes TM band gap between bands 4 and 5 using field-intensity-driven
    pixel updates with smooth proxy minimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 100  # Leave 20s buffer for safety
    
    # --- PWE Solver ---
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/epsilon(r)."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs
    
    def get_G_vectors(n_pw):
        """Returns G vectors sorted by magnitude."""
        G_list = []
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
        return G_list[:n_pw * n_pw]
    
    def fft_index(g, N):
        """Map integer G component to FFT array index."""
        return g if g >= 0 else N + g
    
    def solve_band_structure(cell, kx, ky, n_pw=7, return_eigvecs=False):
        """Solves TM band structure at k=(kx,ky)."""
        fft_coeffs = get_fourier_coeffs(cell)
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Build coupling matrix
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            for j in range(n):
                gx_j, gy_j = G_vectors[j]
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                idx_x = fft_index(Gx_diff, size)
                idx_y = fft_index(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        K_sq = np.zeros(n)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
        
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
    
    def compute_field_intensities(cell, k_points, band_idx, n_pw=7):
        """
        Computes averaged local field intensity for a given band
        across multiple k-points. Returns 2D array (size, size).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(cell, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            vec = eigvecs[:, band_idx]
            
            for px in range(size):
                for py in range(size):
                    rx = px / size
                    ry = py / size
                    
                    field = 0.0
                    for i in range(n):
                        gx_i, gy_i = G_vectors[i]
                        Kx = kx + gx_i
                        Ky = ky + gy_i
                        phase = 2 * np.pi * (Kx * rx + Ky * ry)
                        field += vec[i] * np.exp(1j * phase)
                    
                    total_intensity[px, py] += abs(field)**2
                    count += 1
        
        if count > 0:
            total_intensity /= count
        return total_intensity
    
    def estimate_gap_ratio(cell, k_points, n_pw=7):
        """
        Estimates the gap-midgap ratio between bands 4 and 5.
        Returns 2*(min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky, n_pw)
            if len(eigvals) < 5:
                continue
            w4 = np.sqrt(max(0, eigvals[3]))
            w5 = np.sqrt(max(0, eigvals[4]))
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf') or max_omega_4 == 0:
            return -1.0
        
        ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return ratio
    
    # --- Symmetry handling ---
    
    def fill_symmetric(unique_mask):
        """Fills the full 32x32 cell from a 16x16 unique mask."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i, j]
                coords = [
                    (i, j),
                    (31-i, j),
                    (i, 31-j),
                    (31-i, 31-j),
                    (j, i),
                    (31-j, i),
                    (j, 31-i),
                    (31-j, 31-i)
                ]
                for r, c in coords:
                    cell[r, c] = val
        return cell
    
    def get_unique_pixel_indices():
        """Returns mapping from unique pixel to all symmetric positions."""
        mapping = {}
        for i in range(16):
            for j in range(16):
                coords = [
                    (i, j),
                    (31-i, j),
                    (i, 31-j),
                    (31-i, 31-j),
                    (j, i),
                    (31-j, i),
                    (j, 31-i),
                    (31-j, 31-i)
                ]
                unique_coords = list(set(coords))
                mapping[(i, j)] = unique_coords
        return mapping
    
    # --- K-points for surrogate ---
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X
        (0.5, 0.5),    # M
        (0.25, 0.0),
        (0.25, 0.25),
        (0.15, 0.15),
        (0.35, 0.15),
        (0.15, 0.35),
        (0.3, 0.2),
        (0.2, 0.3),
        (0.4, 0.25),
        (0.25, 0.4),
    ]
    
    # --- Initialize with random binary mask ---
    np.random.seed(42)
    random.seed(42)
    
    unique_mask = np.random.randint(0, 2, (16, 16), dtype=int)
    cell = fill_symmetric(unique_mask)
    
    current_score = estimate_gap_ratio(cell, k_points)
    best_score = current_score
    best_cell = cell.copy()
    
    pixel_mapping = get_unique_pixel_indices()
    
    # Track recent scores for adaptive step size
    recent_scores = [current_score]
    
    # --- Topological optimization loop ---
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5 (0-indexed: 3 and 4)
        intensity_4 = compute_field_intensities(cell, k_points, 3)
        intensity_5 = compute_field_intensities(cell, k_points, 4)
        
        # Compute pressure field: where band 4 dominates, add dielectric
        # where band 5 dominates, remove dielectric
        pressure = intensity_4 - intensity_5
        
        # Normalize
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Compute average pressure for each unique pixel
        pixel_pressures = {}
        for (i, j), coords in pixel_mapping.items():
            avg_pressure = np.mean(pressure[coords])
            pixel_pressures[(i, j)] = avg_pressure
        
        # Select top K pixels to consider flipping
        K_candidates = min(5, len(pixel_pressures))
        sorted_pixels = sorted(pixel_pressures.items(), key=lambda x: abs(x[1]), reverse=True)
        
        best_new_score = -float('inf')
        best_new_cell = None
        best_new_mask = None
        
        for (i, j), avg_p in sorted_pixels[:K_candidates]:
            # Determine flip direction based on pressure sign
            if avg_p > 0:
                # Band 4 stronger -> add dielectric
                new_val = 1
            else:
                # Band 5 stronger -> remove dielectric
                new_val = 0
            
            # If same as current, try opposite
            if unique_mask[i, j] == new_val:
                new_val = 1 - new_val
            
            # Apply change
            old_val = unique_mask[i, j]
            unique_mask[i, j] = new_val
            new_cell = fill_symmetric(unique_mask)
            new_score = estimate_gap_ratio(new_cell, k_points)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_cell = new_cell
                best_new_mask = unique_mask.copy()
            
            # Revert
            unique_mask[i, j] = old_val
        
        # Also try a few random flips to escape local minima
        for _ in range(2):
            ri = random.randint(0, 15)
            rj = random.randint(0, 15)
            old_val = unique_mask[ri, rj]
            unique_mask[ri, rj] = 1 - old_val
            new_cell = fill_symmetric(unique_mask)
            new_score = estimate_gap_ratio(new_cell, k_points)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_cell = new_cell
                best_new_mask = unique_mask.copy()
            
            unique_mask[ri, rj] = old_val
        
        # Accept best candidate
        if best_new_cell is not None and best_new_score > -float('inf'):
            if best_new_score > current_score:
                # Clear improvement
                current_score = best_new_score
                cell = best_new_cell
                unique_mask = best_new_mask
                if current_score > best_score:
                    best_score = current_score
                    best_cell = cell.copy()
            else:
                # Simulated annealing: accept worse with probability
                temp = max(0.1, 1.0 - iteration / 200.0)
                delta = current_score - best_new_score
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_score = best_new_score
                    cell = best_new_cell
                    unique_mask = best_new_mask
                    if current_score > best_score:
                        best_score = current_score
                        best_cell = cell.copy()
        
        recent_scores.append(current_score)
        if len(recent_scores) > 10:
            recent_scores.pop(0)
    
    # Return best cell found
    if best_cell is None:
        best_cell = cell
    
    return best_cell.tolist()
