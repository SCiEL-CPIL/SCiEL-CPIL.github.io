import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands using a hill-climbing algorithm with a 
    fast plane-wave expansion surrogate.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 100  # Leave 20s buffer for the actual evaluation overhead
    
    # --- Surrogate PWE Solver ---
    
    def get_fourier_coeffs(cell):
        """
        Computes the Fourier coefficients of 1/epsilon(r).
        cell is a 2D numpy array of 0s and 1s.
        1 -> dielectric (eps_high), 0 -> air (eps_low).
        We need 1/eps.
        """
        # Create epsilon map
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        
        # Compute FFT
        # The Fourier coefficients are given by:
        # c_G = (1/N^2) * sum_n inv_eps(n) * exp(-i G dot n)
        # This is essentially the normalized FFT.
        fft_coeffs = np.fft.fft2(inv_eps_map)
        fft_coeffs /= (size * size)
        
        # We need to map the FFT output indices to the correct G vectors.
        # np.fft.fft2 puts DC at [0,0], then positive freqs, then negative freqs.
        # We will store them in a dictionary or just use the array and a helper to get G.
        return fft_coeffs, inv_eps_map

    def get_G_indices(n_pw):
        """
        Returns a list of G-vector indices (gx, gy) for a given number of plane waves.
        We select G vectors centered around 0.
        """
        G_indices = []
        # Range of G indices in the FFT array
        # For size N, frequencies are 0, 1, ..., N/2-1, -N/2, ..., -1
        # We want the smallest |G|.
        # Let's just pick a square range of G vectors around 0.
        # The indices in the FFT array for Gx are: 0, 1, ..., N/2, -N/2+1, ..., -1
        # It's easier to generate the list of (gx, gy) integers and then find their index in the FFT array.
        
        # Let's define the set of G vectors as integers.
        # We want to include 0, and the next smallest.
        # For n_pw=7, we want -3..3.
        half = n_pw // 2
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                # Map integer G to FFT array index
                # FFT index for positive G is G.
                # FFT index for negative G is N + G.
                idx_x = gx if gx >= 0 else size + gx
                idx_y = gy if gy >= 0 else size + gy
                G_indices.append((gx, gy, idx_x, idx_y))
        
        # Sort by magnitude to ensure we have the most important ones if we truncate
        G_indices.sort(key=lambda x: x[0]**2 + x[1]**2)
        return G_indices[:n_pw*n_pw]

    def solve_band_structure(cell, kx, ky, n_pw=7):
        """
        Solves the eigenvalue problem for TM modes at k=(kx, ky).
        Returns list of omega^2 values (sorted).
        """
        fft_coeffs, _ = get_fourier_coeffs(cell)
        G_indices = get_G_indices(n_pw)
        
        # Pre-compute the matrix M
        # M_ij = D(G_i - G_j)
        # D(G) is the Fourier coefficient of 1/eps at G.
        
        n = len(G_indices)
        M = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            gx_i, gy_i, idx_x_i, idx_y_i = G_indices[i]
            for j in range(n):
                gx_j, gy_j, idx_x_j, idx_y_j = G_indices[j]
                
                Gx_diff = gx_i - gx_j
                Gy_diff = gy_i - gy_j
                
                # Get index in FFT array
                idx_x_diff = Gx_diff if Gx_diff >= 0 else size + Gx_diff
                idx_y_diff = Gy_diff if Gy_diff >= 0 else size + Gy_diff
                
                # The coefficient is fft_coeffs[idx_x_diff, idx_y_diff]
                # Note: The standard PWE formula uses the coefficient of 1/eps.
                # The matrix element is D(G_i - G_j).
                M[i, j] = fft_coeffs[idx_x_diff, idx_y_diff]
        
        # The eigenvalue problem is:
        # sum_j D(G_i - G_j) E_j = (kx^2 + ky^2 + |K_i|^2) E_i ? 
        # No, the standard form is:
        # (K_i^2) E_i - sum_j D(G_i - G_j) E_j = omega^2 E_i
        # Where K_i = k + G_i.
        
        # Let's construct the matrix A = D - K^2 * I
        # Actually, the equation is:
        # omega^2 E = (K^2) E - D E
        # (D - K^2 I) E = -omega^2 E
        # So we can solve for eigenvalues of (K^2 I - D) and they are omega^2.
        
        K_sq = np.zeros(n)
        for i in range(n):
            gx_i, gy_i, _, _ = G_indices[i]
            Kx = kx + gx_i
            Ky = ky + gy_i
            K_sq[i] = Kx**2 + Ky**2
            
        # Matrix to diagonalize: A = diag(K_sq) - M
        A = np.diag(K_sq) - M
        
        # Solve eigenvalue problem
        # A is Hermitian (since D is Hermitian and K_sq is real)
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            # Sort eigenvalues
            idx_sort = np.argsort(eigvals)
            eigvals = eigvals[idx_sort]
            return eigvals
        except np.linalg.LinAlgError:
            return [0] * n

    def estimate_gap(cell, k_points):
        """
        Estimates the gap between band 4 and 5.
        """
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            eigvals = solve_band_structure(cell, kx, ky, n_pw=7)
            if len(eigvals) < 5:
                continue
            # eigvals are omega^2. We need omega.
            # omega = sqrt(eigval)
            # The bands are sorted.
            # Band 1 is index 0, Band 4 is index 3, Band 5 is index 4.
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        return gap

    # --- Optimization Loop ---
    
    # Define k-points for the surrogate. 
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx+ky <= 1 (in units of 2pi/a)
    # Actually, the standard irreducible BZ is the triangle with vertices (0,0), (1,0), (0.5,0.5).
    # Let's sample a few points.
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.0),    # X (boundary)
        (0.5, 0.5),    # M (corner)
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.1),
        (0.4, 0.1),
    ]
    
    # Initialize cell
    # Start with a random symmetric pattern
    # Symmetry: C4v
    # We only need to define the top-left 16x16 quadrant (since 32 is even, the center lines are i=15, j=15? No, 0..31. Center is between 15 and 16).
    # Reflections:
    # cell[i][j] == cell[31-i][j]
    # cell[i][j] == cell[i][31-j]
    # cell[i][j] == cell[j][i]
    
    # Let's generate a random mask for the unique pixels.
    # Unique pixels are those where i <= 15 and j <= 15? 
    # Due to transpose, cell[i][j] == cell[j][i].
    # Due to reflection, cell[i][j] == cell[31-i][j].
    # The fundamental domain is roughly i in [0, 15], j in [0, 15] with i<=j? 
    # Actually, for even N, the center is at N/2 - 0.5.
    # Indices 0..15 reflect to 31..16.
    # So we just need to define cell[i][j] for i in 0..15, j in 0..15.
    # Then:
    # cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
    # And cell[i][j] = cell[j][i]
    
    cell = np.zeros((size, size), dtype=int)
    for i in range(16):
        for j in range(16):
            val = 1 if random.random() < 0.5 else 0
            cell[i][j] = val
            cell[31-i][j] = val
            cell[i][31-j] = val
            cell[31-i][31-j] = val
            # Transpose symmetry
            cell[j][i] = val
            cell[31-j][i] = val
            cell[j][31-i] = val
            cell[31-j][31-i] = val
            
    # Ensure consistency if i==j or i==31-j etc.
    # The above assignments might conflict if the random values are different for overlapping sets.
    # To be safe, we should generate values for the unique set and then fill.
    
    def fill_symmetric(cell, unique_mask):
        """
        Fills the full cell from a 16x16 unique mask.
        """
        cell[:] = 0
        for i in range(16):
            for j in range(16):
                val = unique_mask[i][j]
                # The 8 symmetric positions
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

    # Regenerate properly
    unique_mask = np.random.randint(0, 2, (16, 16))
    cell = fill_symmetric(cell, unique_mask)
    
    current_gap = estimate_gap(cell, k_points)
    best_gap = current_gap
    best_cell = cell.copy()
    
    temperature = 0.1
    min_temperature = 0.001
    
    # Optimization
    while time.time() - start_time < time_limit:
        # Pick a random unique pixel to flip
        i = random.randint(0, 15)
        j = random.randint(0, 15)
        
        # Flip it in the unique mask
        old_val = unique_mask[i][j]
        unique_mask[i][j] = 1 - old_val
        
        # Fill the full cell
        new_cell = fill_symmetric(cell, unique_mask)
        
        # Evaluate new gap
        new_gap = estimate_gap(new_cell, k_points)
        
        # Accept or reject
        if new_gap > current_gap:
            current_gap = new_gap
            cell = new_cell
            if new_gap > best_gap:
                best_gap = new_gap
                best_cell = new_cell.copy()
        else:
            # Metropolis acceptance
            delta = current_gap - new_gap
            if delta < 0:
                pass # Already accepted above
            else:
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    current_gap = new_gap
                    cell = new_cell
                else:
                    # Revert
                    unique_mask[i][j] = old_val
                    cell = fill_symmetric(cell, unique_mask)
        
        # Cool down
        temperature *= 0.99
        if temperature < min_temperature:
            temperature = min_temperature

    # Convert best_cell to list of lists
    return best_cell.tolist()
