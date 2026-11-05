import numpy as np
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell maximizing the TM band gap between bands 4 and 5.
    Uses a local search algorithm with a fast plane-wave expansion (PWE) surrogate.
    """
    size = problem["pixels"]
    half = size // 2
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard: 220 seconds to be safe
    start_time = time.time()
    deadline = start_time + 220.0
    
    # 1. Initialize the cell
    # Start with the super-ellipse design from the seed
    cell = [[0 for _ in range(size)] for _ in range(size)]
    cx = 7.5
    cy = 7.5
    a = 5.0
    n = 4.0
    
    for i in range(half):
        for j in range(half):
            x = i + 0.5
            y = j + 0.5
            dx = abs(x - cx) / a
            dy = abs(y - cy) / a
            if dx**n + dy**n < 1.0:
                cell[i][j] = 1
                cell[i][size - 1 - j] = 1
                cell[size - 1 - i][j] = 1
                cell[size - 1 - i][size - 1 - j] = 1
    
    # Convert to numpy array for efficiency
    cell_np = np.array(cell, dtype=np.float64)
    
    # 2. Setup PWE parameters
    # Number of G-vectors in each dimension (odd number for symmetry)
    # For fast surrogate, use 7x7 = 49 G-vectors. For final, maybe 15x15.
    # Let's start with 9x9 = 81 G-vectors for a balance.
    N_G = 9 
    G_range = list(range(-N_G//2, N_G//2 + 1))
    G_vectors = np.array([(gx, gy) for gx in G_range for gy in G_range], dtype=np.float64)
    num_G = len(G_vectors)
    
    # Precompute exponentials for Fourier coefficients
    # epsilon_G = 1/N^2 sum_{i,j} epsilon(i,j) exp(-i (gx i + gy j))
    # We will update epsilon_G incrementally.
    
    def compute_fourier_coeffs(cell_arr):
        coeffs = np.zeros(num_G, dtype=np.complex128)
        for idx, (gx, gy) in enumerate(G_vectors):
            # Use matrix multiplication for speed
            # exp(-i * (gx * i + gy * j))
            # i, j are 0..31
            i_idx = np.arange(size)
            j_idx = np.arange(size)
            phase = -1j * (gx * i_idx[:, None] + gy * j_idx[None, :])
            exp_matrix = np.exp(phase)
            coeffs[idx] = np.sum(cell_arr * exp_matrix) / (size * size)
        return coeffs

    # Initial Fourier coefficients
    eps_coeffs = compute_fourier_coeffs(cell_np)
    
    # 3. Define the band structure evaluation
    # We evaluate at several k-points in the irreducible Brillouin zone.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= 1/2 (in units of 2pi/a, where a=1 pixel? No, a=32 pixels? 
    # Usually k is in units of 2pi/L where L is the lattice constant. Here L=32.
    # G vectors are 2pi * (gx, gy) / 32.
    # Let's work in units where G = (2pi/32) * (gx, gy).
    # k points: Gamma (0,0), M (1/2, 0) -> (1/2, 1/2) in reduced coords? 
    # Standard reduced coords: Gamma (0,0), X (1/2, 0), M (1/2, 1/2).
    # Irreducible wedge: 0 <= kx <= ky, kx+ky <= 1/2? No, square lattice irreducible zone is 1/8 of the BZ.
    # Points: Gamma (0,0), M (1/2, 1/2), X (1/2, 0).
    # Path: Gamma -> M -> X -> Gamma?
    # Let's sample points:
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.25, 0.25),    # Interior
        (0.5, 0.5),      # M
        (0.5, 0.0),      # X
        (0.25, 0.0),     # Edge
        (0.1, 0.1),
        (0.4, 0.2)
    ]
    
    def compute_bands(eps_coeffs, k_points):
        """
        Computes the lowest 6 bands at given k-points.
        Returns (max_omega_4, min_omega_5) over all k-points.
        """
        max_w4 = 0.0
        min_w5 = float('inf')
        
        factor = 2.0 * np.pi / size
        # Precompute k+G vectors
        kG_vectors_list = []
        for kx, ky in k_points:
            k_vec = np.array([kx, ky]) * factor
            kG = k_vec[None, :] + G_vectors * factor
            kG_vectors_list.append(kG)
        
        for kG in kG_vectors_list:
            # Build Hamiltonian matrix H
            # H_{GG'} = (k+G) dot (k+G') * epsilon_{G-G'}
            # Note: epsilon_{G-G'} requires mapping G indices to G vectors.
            # Since G_vectors is a grid, G - G' is also in the grid if the grid is closed under subtraction?
            # With N_G=9, indices -4 to 4. Difference can be -8 to 8.
            # We only have coeffs for -4 to 4. We need to handle aliasing or use a larger grid for coeffs.
            # For a 32x32 pixel cell, the Fourier series is finite. 
            # However, the PWE method requires epsilon_{G-G'}.
            # If G-G' is outside our computed range, we must compute it.
            # To avoid this, we should compute epsilon_G for a larger range, or ensure N_G is large enough 
            # such that the product of the truncation error is negligible.
            # Actually, the standard PWE uses a finite set of G. The matrix elements are 
            # H_{G, G'} = (k+G) . (k+G') * epsilon_{G-G'}.
            # We need epsilon_{G-G'} for all G, G' in our set.
            # The set of differences G-G' is larger than the set of G.
            # We should compute epsilon_G for the full range of differences.
            # Max difference: 4 - (-4) = 8. So we need G from -8 to 8.
            # Let's recompute epsilon_coeffs for a larger grid.
            pass
            
        # This approach is getting complicated for incremental updates.
        # Alternative: Just recompute the matrix from scratch for each evaluation.
        # With N_G=9 (81 modes), matrix is 81x81.
        # Eigenvalue of 81x81 is fast (~1ms).
        # Evaluating 7 k-points -> 7ms.
        # Local search step: 1 flip -> 7ms.
        # 220s / 0.007s ~ 30000 steps. This is plenty.
        
        return max_w4, min_w5

    # Let's implement a robust compute_bands that handles the G-G' indices correctly.
    # We will compute epsilon_G for all G in the range [-G_max, G_max] where G_max = N_G//2.
    # But G-G' can be up to N_G-1.
    # So we need epsilon_G for G in [-(N_G-1), N_G-1].
    # Let's set G_max_search = N_G - 1.
    
    def get_epsilon_G(gx, gy, cell_arr):
        """Compute epsilon_G for a specific G vector."""
        i_idx = np.arange(size)
        j_idx = np.arange(size)
        phase = -1j * (gx * i_idx[:, None] + gy * j_idx[None, :])
        exp_matrix = np.exp(phase)
        return np.sum(cell_arr * exp_matrix) / (size * size)

    # Precompute exponentials for all G in the required range for fast updates?
    # Or just recompute the matrix each time.
    
    def evaluate_gap(cell_arr):
        """
        Evaluate the gap ratio for the given cell.
        Returns the ratio (min_w5 - max_w4) / (min_w5 + max_w4).
        """
        # Determine required G range
        # We use N_G=9 for the basis.
        # Differences G-G' range from -(N_G-1) to N_G-1.
        # So we need epsilon_G for gx, gy in [-(N_G-1), N_G-1].
        G_diff_range = list(range(-(N_G-1), N_G+1))
        
        # Compute all necessary epsilon_G
        # This is expensive if done from scratch every time.
        # But we can optimize by caching or using the fact that we only flip one pixel.
        
        # For now, let's just compute from scratch to ensure correctness, then optimize if needed.
        # 32x32 * ~100 G vectors = 100k ops. Very fast.
        
        eps_dict = {}
        for gx in G_diff_range:
            for gy in G_diff_range:
                eps_dict[(gx, gy)] = get_epsilon_G(gx, gy, cell_arr)
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        factor = 2.0 * np.pi / size
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky]) * factor
            kG = k_vec[None, :] + G_vectors * factor
            
            # Build H
            H = np.zeros((num_G, num_G), dtype=np.complex128)
            for i, G1 in enumerate(G_vectors):
                for j, G2 in enumerate(G_vectors):
                    G_diff = (G1[0] - G2[0], G1[1] - G2[1])
                    eps_val = eps_dict[G_diff]
                    dot_prod = np.dot(kG[i], kG[j])
                    H[i, j] = dot_prod * eps_val
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigenvalues, _ = np.linalg.eigh(H)
            
            # Eigenvalues are omega^2
            # We want the 4th and 5th bands (indices 3 and 4)
            # Ensure we have at least 5 bands
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5
        
        if min_w5 == float('inf') or max_w4 == 0:
            return -1.0
            
        ratio = (min_w5 - max_w4) / (min_w5 + max_w4)
        return ratio

    # Local Search
    # We will iterate over the 8x8 quadrant.
    # For each pixel, try flipping it.
    # Keep track of the best cell.
    
    best_cell = cell_np.copy()
    best_ratio = evaluate_gap(best_cell)
    
    # To speed up, we can precompute the exponential matrices for each pixel in the quadrant
    # and update eps_dict incrementally.
    
    # Let's implement incremental updates.
    # eps_G = 1/N^2 sum_{i,j} eps(i,j) exp(-i G . r)
    # Flip pixel (i,j) in quadrant -> affects 4 pixels in full cell due to symmetry.
    # Delta eps_G = sum_{sym} +/- 1/N^2 exp(-i G . r_sym)
    
    # Precompute phase factors for the 4 symmetric positions for each G in the diff range.
    # This is 100 G vectors * 4 positions = 400 complex numbers.
    
    # Let's stick to the simple recompute method first to ensure it works, as 30000 evals is fast enough.
    
    # Optimized Local Search
    # Shuffle the quadrant pixels to avoid bias.
    quadrant_pixels = [(i, j) for i in range(half) for j in range(half)]
    np.random.shuffle(quadrant_pixels)
    
    no_improvement_count = 0
    max_no_improvement = 1000 # Reset if stuck
    
    while time.time() < deadline - 10.0: # Stop 10s before deadline for final check
        improved = False
        for i, j in quadrant_pixels:
            if time.time() > deadline - 10.0:
                break
            
            # Flip the pixel
            old_val = best_cell[i, j]
            new_val = 1.0 - old_val
            
            # Apply symmetry
            best_cell[i, j] = new_val
            best_cell[i, size - 1 - j] = new_val
            best_cell[size - 1 - i, j] = new_val
            best_cell[size - 1 - i, size - 1 - j] = new_val
            
            # Evaluate
            current_ratio = evaluate_gap(best_cell)
            
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                improved = True
                no_improvement_count = 0
            else:
                # Revert
                best_cell[i, j] = old_val
                best_cell[i, size - 1 - j] = old_val
                best_cell[size - 1 - i, j] = old_val
                best_cell[size - 1 - i, size - 1 - j] = old_val
                no_improvement_count += 1
        
        if not improved:
            # Shuffle again to escape local minima
            np.random.shuffle(quadrant_pixels)
            if no_improvement_count > max_no_improvement:
                # Try a random restart or larger move?
                # For now, just continue shuffling
                no_improvement_count = 0
                # Occasionally, try a 2-pixel flip or random perturbation
                # Simple random perturbation
                for _ in range(5):
                    ri, rj = np.random.randint(0, half), np.random.randint(0, half)
                    old_val = best_cell[ri, rj]
                    new_val = 1.0 - old_val
                    best_cell[ri, rj] = new_val
                    best_cell[ri, size - 1 - rj] = new_val
                    best_cell[size - 1 - ri, rj] = new_val
                    best_cell[size - 1 - ri, size - 1 - rj] = new_val
                    
                    current_ratio = evaluate_gap(best_cell)
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                    else:
                        best_cell[ri, rj] = old_val
                        best_cell[ri, size - 1 - rj] = old_val
                        best_cell[size - 1 - ri, rj] = old_val
                        best_cell[size - 1 - ri, size - 1 - rj] = old_val

    # Convert back to list of lists
    result_cell = [[int(round(cell_np[i][j])) for j in range(size)] for i in range(size)]
    
    return result_cell
