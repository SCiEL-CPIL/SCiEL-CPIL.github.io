import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _get_symmetry_indices(size):
    """
    Returns a list of unique pixel indices under C4v symmetry (square symmetry group).
    For a pixel (i, j), its symmetric equivalents are:
    (i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j)
    (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)
    """
    size = 32
    unique_indices = []
    seen = set()
    for i in range(size):
        for j in range(size):
            # Generate all 8 symmetric positions
            sym_pos = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i)
            ]
            # Check if any of these positions have already been seen
            if not any(pos in seen for pos in sym_pos):
                unique_indices.append(sym_pos)
                for pos in sym_pos:
                    seen.add(pos)
    return unique_indices

def _fast_surrogate_gap(cell_flat):
    """
    Fast surrogate model for the band gap.
    Uses a coarse grid and reduced plane waves.
    cell_flat: 1D array of 0/1 values of length 32*32.
    Returns estimated gap-midgap ratio.
    """
    N = 32
    cell = cell_flat.reshape(N, N)
    
    # Coarse grid for surrogate (e.g., 8x8)
    coarse_size = 8
    coarse_cell = np.zeros((coarse_size, coarse_size))
    block_size = N // coarse_size
    for i in range(coarse_size):
        for j in range(coarse_size):
            block = cell[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            coarse_cell[i, j] = np.mean(block)
            
    # Downsampled dielectric function
    eps_low = 1.0
    eps_high = 12.0
    eps_coarse = eps_low + (eps_high - eps_low) * coarse_cell
    
    # Plane Wave Method Parameters
    # Number of plane waves in each direction
    N_pw = 11 
    # Total plane waves
    N_tot = N_pw * N_pw
    
    # Reciprocal lattice vectors
    # For square lattice a=1
    G = []
    for nx in range(-N_pw//2, N_pw//2 + 1):
        for ny in range(-N_pw//2, N_pw//2 + 1):
            G.append((2 * np.pi * nx, 2 * np.pi * ny))
    G = np.array(G)
    G2 = G[:, 0]**2 + G[:, 1]**2
    
    # Fourier coefficients of 1/eps
    # We need inv_eps = 1 / eps
    inv_eps_coarse = 1.0 / eps_coarse
    
    # Calculate Fourier coefficients of inv_eps
    # k = (nx, ny)
    # F(k) = sum_{i,j} inv_eps(i,j) * exp(-i 2 pi (kx i + ky j) / N_coarse)
    # Using FFT for efficiency
    # Note: FFT gives unnormalized sum. We need to be careful with scaling.
    # For discrete FFT: F(k) = sum_{n} f(n) exp(-i 2 pi kn/N)
    # Our G vectors are 2 pi * integer.
    # So we just need the FFT of inv_eps_coarse.
    
    # Reshape inv_eps to 2D
    # FFT
    F_eps = np.fft.fft2(inv_eps_coarse)
    
    # Map G indices to FFT indices
    # FFT index (kx, ky) corresponds to frequency (kx, ky) if kx < N/2, else kx - N
    # Our G vectors are in range [-N_pw/2, N_pw/2]
    # FFT indices are in [0, N_coarse-1]
    
    # We need to construct the matrix E(G, G') = delta(G-G') + F(G-G') / eps(G)
    # Actually, the standard PWM equation is:
    # sum_{G'} [ (K+G)^2 delta_{G,G'} + (1/eps)_{G-G'} ] A_{G'} = omega^2 A_G
    # Where (1/eps)_k = F(1/eps)_k
    
    # Let's build the matrix H
    # H_{G, G'} = (K+G)^2 delta_{G,G'} + F(1/eps)_{G-G'}
    
    # K points in the irreducible Brillouin zone
    # For square lattice, IBZ is 0 <= ky <= kx <= pi
    # We sample a few K points
    K_points = [
        (0.0, 0.0),      # Gamma
        (np.pi/2, 0.0),  # M
        (np.pi/2, np.pi/2), # X
        (np.pi/4, 0.0),  # Interior
        (np.pi/4, np.pi/4) # Interior
    ]
    
    # Precompute F(1/eps) for all G differences
    # G_diff = G_i - G_j
    # We need F(G_diff)
    
    # To speed up, we precompute the FFT matrix mapping
    # F(k) for k in G
    # We can just compute the full FFT and index into it
    
    # Let's create a lookup for F(G)
    # G values are integers * 2pi. FFT frequencies are integers.
    # So F(G) is just F_fft[nx, ny]
    
    max_n = N_pw // 2
    # Create a 2D array of F values
    F_vals = np.zeros((N_pw, N_pw), dtype=complex)
    for idx, (gx, gy) in enumerate(G):
        nx = int(gx / (2 * np.pi))
        ny = int(gy / (2 * np.pi))
        # Map to FFT index
        idx_x = nx % coarse_size
        idx_y = ny % coarse_size
        F_vals[nx + max_n, ny + max_n] = F_eps[idx_x, idx_y]
        
    # Note: The FFT normalization. 
    # np.fft.fft2 returns sum_{n} f(n) exp(-i 2 pi kn/N).
    # The coefficient in the expansion f(x) = sum_k F_k exp(i 2 pi k x / N) is F_k / N^2.
    # In the PWM equation, the term is (1/eps)_k = (1/N^2) * F_k (Fourier series coeff).
    # So we should divide F_vals by coarse_size**2.
    F_vals /= coarse_size**2
    
    # Now build the eigenvalue problem for each K
    # We want the 4th and 5th bands.
    # We will solve for the lowest, say, 6 bands.
    
    max_gap = 0.0
    
    for Kx, Ky in K_points:
        K = np.array([Kx, Ky])
        
        # Construct matrix H (N_tot x N_tot)
        H = np.zeros((N_tot, N_tot), dtype=complex)
        
        for i, Gi in enumerate(G):
            Ki = K + Gi
            Ki2 = Ki[0]**2 + Ki[1]**2
            
            # Diagonal term
            H[i, i] = Ki2
            
            # Off-diagonal terms
            for j, Gj in enumerate(G):
                if i == j:
                    continue
                G_diff = Gi - Gj
                nx_diff = int(G_diff[0] / (2 * np.pi))
                ny_diff = int(G_diff[1] / (2 * np.pi))
                
                # Index into F_vals
                idx_x = nx_diff + max_n
                idx_y = ny_diff + max_n
                H[i, j] = F_vals[idx_x, idx_y]
                
        # Solve eigenvalue problem
        # H is Hermitian? 
        # F(-k) = conj(F(k)) if f is real. 
        # F_vals should satisfy this.
        # Let's ensure H is Hermitian by symmetrizing
        H = (H + H.conj().T) / 2
        
        # Compute eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(H)
            # Sort eigenvalues
            eigvals.sort()
            
            # Take the first 6 bands
            # Band 4 is index 3, Band 5 is index 4 (0-indexed)
            # We want gap between 4 and 5
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                
                # Gap-midgap ratio
                if omega_4 > 0:
                    gap_ratio = 2 * (omega_5 - omega_4) / (omega_5 + omega_4)
                else:
                    gap_ratio = 0.0
                    
                if gap_ratio > max_gap:
                    max_gap = gap_ratio
                    
        except np.linalg.LinAlgError:
            pass
            
    return max_gap

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Symmetry indices
    sym_indices = _get_symmetry_indices(size)
    num_unique_pixels = len(sym_indices)
    
    # Initial design: A dielectric square with a central air hole
    # This is a common structure for higher-order gaps
    cell = np.zeros((size, size), dtype=int)
    
    # Fill with dielectric (1)
    cell[:] = 1
    
    # Create a central air hole (0)
    # Let's try a square hole of size 8x8 in the center
    hole_size = 8
    center = size // 2
    cell[center - hole_size//2 : center + hole_size//2, center - hole_size//2 : center + hole_size//2] = 0
    
    # Flatten
    cell_flat = cell.flatten()
    
    # Calculate initial gap
    current_gap = _fast_surrogate_gap(cell_flat)
    best_cell = cell_flat.copy()
    best_gap = current_gap
    
    # Local search
    # We will flip pixels in the unique symmetry group
    # Random restarts if stuck
    
    max_iterations = 5000
    iter_count = 0
    
    # Use a simple hill climbing with random restarts
    # If no improvement for X steps, restart from a new random symmetric design
    
    no_improve_count = 0
    restart_threshold = 50
    
    while time.time() < deadline and iter_count < max_iterations:
        # Pick a random unique pixel group
        idx = random.randint(0, num_unique_pixels - 1)
        positions = sym_indices[idx]
        
        # Flip these pixels
        new_cell = best_cell.copy()
        for pos in positions:
            i, j = pos
            flat_idx = i * size + j
            new_cell[flat_idx] = 1 - new_cell[flat_idx]
            
        # Evaluate
        new_gap = _fast_surrogate_gap(new_cell)
        
        if new_gap > best_gap:
            best_gap = new_gap
            best_cell = new_cell
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        iter_count += 1
        
        # Restart if stuck
        if no_improve_count > restart_threshold:
            # Create a new random symmetric design
            # Bias towards dielectric or air?
            # Let's try a few templates
            
            # Template 1: Random noise
            if random.random() < 0.5:
                cell_new = np.zeros((size, size), dtype=int)
                for pos_group in sym_indices:
                    val = random.randint(0, 1)
                    for pos in pos_group:
                        i, j = pos
                        cell_new[i, j] = val
            else:
                # Template 2: Dielectric with random holes
                cell_new = np.ones((size, size), dtype=int)
                for pos_group in sym_indices:
                    if random.random() < 0.3: # 30% chance to be air
                        val = 0
                        for pos in pos_group:
                            i, j = pos
                            cell_new[i, j] = val
            
            cell_new_flat = cell_new.flatten()
            new_gap = _fast_surrogate_gap(cell_new_flat)
            
            if new_gap > best_gap:
                best_gap = new_gap
                best_cell = cell_new_flat
                no_improve_count = 0
            else:
                # Try a specific "split rod" or "cross" structure
                # Cross shape
                cell_cross = np.zeros((size, size), dtype=int)
                center = size // 2
                arm_width = 4
                # Horizontal bar
                cell_cross[center - arm_width//2 : center + arm_width//2, :] = 1
                # Vertical bar
                cell_cross[:, center - arm_width//2 : center + arm_width//2] = 1
                # Add some noise
                for pos_group in sym_indices:
                    if random.random() < 0.1:
                        for pos in pos_group:
                            i, j = pos
                            cell_cross[i, j] = 1 - cell_cross[i, j]
                
                cell_cross_flat = cell_cross.flatten()
                gap_cross = _fast_surrogate_gap(cell_cross_flat)
                
                if gap_cross > best_gap:
                    best_gap = gap_cross
                    best_cell = cell_cross_flat
                    no_improve_count = 0
                else:
                    no_improve_count = 0 # Reset to continue searching from current best
                    
    # Convert best_cell back to nested list
    best_cell_2d = best_cell.reshape(size, size)
    result = [list(row) for row in best_cell_2d]
    
    return result

# EVOLVE-BLOCK-END
