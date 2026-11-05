import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using density-based topological optimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 105  # Leave 15s buffer
    
    # --- PWE Solver ---
    
    def get_fourier_coeffs_periodic(density):
        """
        Computes the Fourier coefficients of 1/epsilon(r) using periodic wrapping.
        density is a 2D numpy array of values in [0,1] representing dielectric fraction.
        """
        # Map density to epsilon: eps = eps_low + (eps_high - eps_low) * density
        eps_map = eps_low + (eps_high - eps_low) * density
        inv_eps_map = 1.0 / eps_map
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        return fft_coeffs, inv_eps_map

    def get_G_vectors(n_pw):
        """
        Returns G vectors (as integer multiples of 2pi/a) sorted by magnitude.
        """
        G_list = []
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        G_list.sort(key=lambda x: x[0]**2 + x[1]**2)
        return G_list[:n_pw * n_pw]

    def fft_index_periodic(g, N):
        """Map integer G component to FFT array index with periodic wrapping."""
        return g % N

    def solve_band_structure(density, kx, ky, n_pw=7, return_eigvecs=False):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns sorted omega^2 values, and optionally eigenvectors.
        """
        fft_coeffs, _ = get_fourier_coeffs_periodic(density)
        G_vectors = get_G_vectors(n_pw)
        n = len(G_vectors)
        
        # Build matrix M where M[i,j] = D(G_i - G_j)
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            for j in range(n):
                gx_j, gy_j = G_vectors[j]
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                idx_x = fft_index_periodic(Gx_diff, size)
                idx_y = fft_index_periodic(Gy_diff, size)
                M[i, j] = fft_coeffs[idx_x, idx_y]
        
        # K^2 values
        K_sq = np.zeros(n)
        for i in range(n):
            gx_i, gy_i = G_vectors[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
        
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
                return [0] * n, np.zeros((n, n), dtype=complex)
            return [0] * n

    def compute_field_intensities(density, k_points, band_idx, n_pw=7):
        """
        Computes the local field intensity at each pixel for a given band
        across multiple k-points, averaged.
        Returns a 2D array of shape (size, size).
        """
        total_intensity = np.zeros((size, size))
        count = 0
        
        for kx, ky in k_points:
            eigvals, eigvecs = solve_band_structure(density, kx, ky, n_pw, return_eigvecs=True)
            if len(eigvals) <= band_idx:
                continue
            
            # Get eigenvector for the specified band
            vec = eigvecs[:, band_idx]
            
            # Compute field at each pixel
            G_vectors = get_G_vectors(n_pw)
            n = len(G_vectors)
            
            for px in range(size):
                for py in range(size):
                    # Position in units of lattice constant: r = (px/size, py/size)
                    rx = px / size
                    ry = py / size
                    
                    # Compute field
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

    def estimate_gap(density, k_points, n_pw=7):
        """
        Estimates the gap between band 4 and 5.
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(density, kx, ky, n_pw)
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

    # --- Symmetry handling for final output ---
    
    def fill_symmetric_binary(unique_mask):
        """
        Fills the full 32x32 cell from a 16x16 unique binary mask.
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
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
                    cell[r][c] = val
        return cell

    def symmetrize_density(density):
        """
        Enforces square symmetry on a density array by averaging over symmetry group.
        """
        sym_density = np.zeros_like(density)
        for i in range(size):
            for j in range(size):
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
                sym_density[i, j] = np.mean([density[r, c] for r, c in unique_coords])
        return sym_density

    # --- Optimization ---
    
    # Define k-points for the surrogate
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
        (0.15, 0.05),
        (0.35, 0.15),
    ]
    
    # Initialize with a random smooth density
    random.seed(42)
    density = np.random.uniform(0.2, 0.8, (size, size))
    
    # Apply initial smoothing
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=2.0)
    density = np.clip(density, 0.05, 0.95)
    
    # Enforce symmetry
    density = symmetrize_density(density)
    
    current_gap = estimate_gap(density, k_points)
    best_gap = current_gap
    best_density = density.copy()
    
    # Density-based topological optimization loop
    # Use field intensities to guide density updates
    num_iterations = 150
    learning_rate = 0.1
    
    for iteration in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute field intensities for bands 4 and 5
        intensity_4 = compute_field_intensities(density, k_points, 3)  # Band 4 (0-indexed: 3)
        intensity_5 = compute_field_intensities(density, k_points, 4)  # Band 5 (0-indexed: 4)
        
        # Strategy: 
        # - Where band 4 has high intensity and band 5 has low intensity: increase density (more dielectric)
        #   This raises band 5 more than band 4, potentially opening the gap
        # - Where band 5 has high intensity and band 4 has low intensity: decrease density (less dielectric)
        #   This lowers band 5 more than band 4, potentially opening the gap
        
        # Compute a "pressure" field: positive means increase density, negative means decrease
        pressure = intensity_4 - intensity_5
        
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max > 1e-10:
            pressure /= p_max
        
        # Update density: increase where pressure is positive, decrease where negative
        # Use a sigmoid-like update to keep values in [0,1]
        delta_density = learning_rate * pressure
        density = density + delta_density
        density = np.clip(density, 0.05, 0.95)
        
        # Apply smoothing to prevent checkerboard patterns
        density = gaussian_filter(density, sigma=1.0)
        density = np.clip(density, 0.05, 0.95)
        
        # Enforce symmetry
        density = symmetrize_density(density)
        
        new_gap = estimate_gap(density, k_points)
        
        # Accept if better, or with some probability if worse (simulated annealing)
        if new_gap > current_gap:
            current_gap = new_gap
            if new_gap > best_gap:
                best_gap = new_gap
                best_density = density.copy()
        else:
            # Reject and revert with some probability
            if random.random() > 0.3:
                density = best_density.copy()
                current_gap = best_gap
            # Otherwise keep the worse solution to explore
        
        # Occasionally do a random perturbation to escape local minima
        if iteration % 15 == 14:
            perturbation = np.random.uniform(-0.1, 0.1, (size, size))
            density = density + perturbation
            density = np.clip(density, 0.05, 0.95)
            density = gaussian_filter(density, sigma=1.0)
            density = symmetrize_density(density)
    
    # Final: threshold the density to get binary pixels
    best_density = np.clip(best_density, 0, 1)
    threshold = 0.5
    binary_cell = (best_density >= threshold).astype(int)
    
    # Enforce exact symmetry on the binary cell
    final_cell = np.zeros((size, size), dtype=int)
    for i in range(16):
        for j in range(16):
            val = binary_cell[i][j]
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
                final_cell[r][c] = val
    
    return final_cell.tolist()
