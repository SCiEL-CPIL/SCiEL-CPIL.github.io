"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Represent the cell via C4v-symmetric wavelet detail coefficients.
    2. Optimize detail coefficients to create fractal-like textures.
    3. Use fast plane-wave evaluation with C4v symmetry enforcement.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- C4v Symmetry Enforcement ---
    def enforce_symmetry(c):
        """Enforce C4v symmetry by averaging over all 8 symmetric points."""
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    # --- Wavelet-based Representation ---
    # Use 2D Haar wavelet decomposition. For C4v symmetry, we only need
    # independent coefficients in the fundamental domain.
    # The fundamental domain for C4v on a 32x32 grid is roughly the top-left
    # 16x16 triangle (i <= j, i < 16).
    
    # Precompute the list of independent wavelet detail coefficients
    # For a 32x32 image, we have 5 levels of decomposition.
    # At each level, we have horizontal, vertical, and diagonal details.
    # Due to C4v symmetry, many of these are related.
    
    # Simpler approach: work directly with the pixel values in the fundamental domain
    # and reconstruct the full symmetric cell.
    
    def fundamental_domain_pixels():
        """Return indices of pixels in the fundamental domain of C4v."""
        indices = []
        for i in range(size // 2):
            for j in range(i, size // 2):
                indices.append((i, j))
        return indices
    
    fd_indices = fundamental_domain_pixels()
    n_fd = len(fd_indices)
    
    def fd_to_full(fd_values):
        """Reconstruct full symmetric cell from fundamental domain values."""
        cell = np.zeros((size, size), dtype=float)
        for idx, val in zip(fd_indices, fd_values):
            i, j = idx
            # Place in all 8 symmetric positions
            cell[i, j] = val
            cell[size-1-i, j] = val
            cell[i, size-1-j] = val
            cell[size-1-i, size-1-j] = val
            cell[j, i] = val
            cell[size-1-j, j] = val
            cell[j, size-1-i] = val
            cell[size-1-j, size-1-i] = val
        # Threshold to binary
        cell = (cell > 0.5).astype(int)
        return cell

    # --- Fast Plane-Wave Evaluator ---
    # Use 9x9 plane waves (81 modes) for better accuracy
    pw_half = 4
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=float)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=float)
    
    # Precompute G differences modulo size
    G_diff_idx_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_idx_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_idx_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diff_idx_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size
    
    # K-points in irreducible BZ
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # Along Gamma-M: (0, t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    # Along M-X: (t, 0.5-t)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    # X (0.5, 0.5)
    k_points.append((0.5, 0.5))
    # Additional interior points for better coverage
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((t, t))
        k_points.append((0.1, 0.4 - t))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        # Build inverse epsilon grid
        inv_eps = np.where(cell_grid == 1, 1.0/eps_high, 1.0/eps_low)
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = (H + H.conj().T) * 0.5  # Ensure Hermitian
            
            # Get eigenvalues (we need indices 3 and 4 for bands 4 and 5)
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0.0, w4_sq))
                w5 = np.sqrt(max(0.0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- Optimization: Optimize fundamental domain values ---
    # Initialize with a pattern that has some structure
    # Start with a checkerboard-like pattern in the fundamental domain
    fd_values = np.zeros(n_fd)
    for idx, (i, j) in enumerate(fd_indices):
        # Create a pattern with alternating regions
        if (i // 4 + j // 4) % 2 == 0:
            fd_values[idx] = 1.0
        else:
            fd_values[idx] = 0.0
    
    current_cell = fd_to_full(fd_values)
    current_score = evaluate_gap(current_cell)
    best_cell = current_cell.copy()
    best_score = current_score
    
    # Gradient-free optimization using coordinate descent with random restarts
    temperature = 0.3
    cooling_rate = 0.999
    max_iterations = 30000
    iteration = 0
    
    # Precompute orbit info for each fundamental domain pixel
    # For each fd pixel, which full pixels does it affect?
    def get_orbit(i, j):
        """Get all 8 symmetric positions for pixel (i,j)."""
        pts = [
            (i, j),
            (size-1-i, j),
            (i, size-1-j),
            (size-1-i, size-1-j),
            (j, i),
            (size-1-j, j),
            (j, size-1-i),
            (size-1-j, size-1-i)
        ]
        # Remove duplicates
        return list(set(pts))
    
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Pick a random fundamental domain pixel to modify
        idx = np.random.randint(n_fd)
        i, j = fd_indices[idx]
        
        # Try flipping this pixel's value
        new_fd_values = fd_values.copy()
        new_fd_values[idx] = 1.0 - new_fd_values[idx]
        
        new_cell = fd_to_full(new_fd_values)
        new_score = evaluate_gap(new_cell)
        
        # Accept or reject based on Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            fd_values = new_fd_values
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
    
    # Finalize
    final_cell = enforce_symmetry(best_cell)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
