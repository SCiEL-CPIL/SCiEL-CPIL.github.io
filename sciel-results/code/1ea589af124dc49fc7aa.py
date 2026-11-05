import time
import math
import numpy as np

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin for 230s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # 1. Setup PWE Solver
    # We use a plane-wave expansion with N_k waves.
    # For a 32x32 grid, the Fourier components G range from -15 to 15.
    # Using all 31x31 = 961 waves is too slow for iterative optimization in Python.
    # We truncate to a smaller set, e.g., G_x, G_y in [-6, 6] -> 13x13 = 169 waves.
    # This is an approximation, but consistent for relative comparison.
    
    N_g = 7 # Range -6 to 6
    G_values = np.arange(-N_g, N_g + 1)
    G_vectors = [(gx, gy) for gx in G_values for gy in G_values]
    N_pwe = len(G_vectors)
    
    # Precompute G^2 and G dot K
    G_sq = np.array([gx*gx + gy*gy for gx, gy in G_vectors], dtype=float)
    
    def get_fourier_coeffs(cell):
        """
        Compute Fourier coefficients of 1/epsilon(x,y) for the given cell.
        cell is a 32x32 array of 0/1.
        eps = 1 if 0 (air), 12 if 1 (dielectric).
        We need 1/eps.
        """
        # 1/eps map
        # air: 1/1 = 1
        # dielectric: 1/12
        inv_eps = np.ones((size, size), dtype=float)
        inv_eps[cell == 1] = 1.0 / eps_high
        
        # Compute DFT of inv_eps
        # The plane wave method uses coefficients:
        # 1/eps(G) = (1/N^2) * sum_{x,y} (1/eps(x,y)) * exp(-i G . r)
        # Since we are in a discrete pixel grid, we approximate the integral by sum.
        # G = 2*pi * (nx, ny) / L, where L=32 pixels.
        # The discrete sum corresponds to FFT.
        
        # We need coefficients for G in G_vectors.
        # G_vectors are in units of 2*pi/L.
        # So we take the FFT of inv_eps and pick the indices corresponding to G_vectors.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        # Normalize by N^2
        fft_inv_eps /= (size * size)
        
        coeffs = np.zeros(N_pwe, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            # FFT index mapping:
            # np.fft.fft2 gives frequencies 0, 1, ..., N/2, -N/2+1, ..., -1
            # Our gx, gy are in -N_g..N_g
            # We need to map gx to the correct FFT bin.
            # If gx >= 0: bin = gx
            # If gx < 0: bin = size + gx
            ix = gx % size
            iy = gy % size
            coeffs[idx] = fft_inv_eps[ix, iy]
            
        return coeffs

    def solve_bands(cell, kx, ky):
        """
        Solve the eigenvalue problem for a given k-point.
        Returns the sorted list of omega^2 (or omega, we can compare squared).
        """
        # Get 1/eps coefficients
        d_inv_eps = get_fourier_coeffs(cell)
        
        # Build the matrix A
        # A_{mn} = delta_{mn} * |k + G_m|^2 * (1/eps(0)) + (1/eps(G_m - G_n)) * delta_{m-n, 0? No.}
        
        # The equation is:
        # sum_n [ |k+G_m|^2 delta_{mn} / eps(0) + (1/eps(G_m - G_n)) ] E_n = omega^2 E_m
        # Wait, the standard form is:
        # sum_n [ delta_{mn} (k+G_m)^2 / eps(0) + 1/eps(G_m - G_n) ] E_n = omega^2 E_m
        # Actually, the term 1/eps(G) multiplies the field.
        # The matrix is M_{mn} = (k+G_m)^2 * delta_{mn} * (1/eps(0)) + (1/eps(G_m - G_n))
        # This is not quite right. The 1/eps is a convolution.
        # In Fourier space, the convolution becomes a multiplication? No, 1/eps is periodic.
        # The matrix elements are:
        # M_{mn} = (k+G_m)^2 * delta_{mn} * (1/eps(0)) + (1/eps(G_m - G_n))
        # Wait, the term (1/eps) is expanded in Fourier series: 1/eps(r) = sum_G (1/eps)_G e^{iG.r}
        # The eigenvalue equation is:
        # sum_G' (k+G+G')^2 (1/eps)_{-G'} E_{G'} = omega^2 E_G
        # Let n correspond to G'. Then m corresponds to G.
        # M_{mn} = (k + G_m + G_n)^2 * (1/eps)_{-(G_m+G_n)} ? No.
        
        # Let's stick to the standard form found in literature (e.g. Joannopoulos):
        # sum_{G'} [ (k+G)^2 delta_{G,G'} / eps(0) + 1/eps(G-G') ] E_{G'} = omega^2 E_G
        # This assumes eps(0) is the average.
        # Actually, the term is:
        # M_{mn} = (k+G_m)^2 * delta_{mn} * (1/eps(0)) + (1/eps)_{G_m - G_n}
        
        # Let's precompute the difference indices
        # For each m, n, we need the coefficient of 1/eps at G_m - G_n.
        # We have coeffs for G in G_vectors.
        # If G_m - G_n is not in G_vectors, we assume it's 0 (truncation error).
        
        # To speed up, we can precompute the matrix structure for the 1/eps part?
        # No, 1/eps changes every iteration.
        
        # Optimization: 
        # The matrix is N_pwe x N_pwe.
        # N_pwe = 169. 169x169 eigensystem is fast.
        
        # Build M
        # We need (1/eps)_{G_m - G_n}
        # Let's create a lookup dictionary for the coeffs
        coeff_dict = {}
        for idx, (gx, gy) in enumerate(G_vectors):
            coeff_dict[(gx, gy)] = d_inv_eps[idx]
            
        # We also need the zero-order term (G=0,0)
        eps_avg_inv = coeff_dict.get((0,0), 1.0)
        
        M = np.zeros((N_pwe, N_pwe), dtype=complex)
        
        # Precompute k+G vectors
        kG_sq = np.zeros(N_pwe, dtype=float)
        for m, (gx, gy) in enumerate(G_vectors):
            kx_total = kx + 2 * math.pi * gx
            ky_total = ky + 2 * math.pi * gy
            kG_sq[m] = kx_total**2 + ky_total**2
            
        # Fill M
        for m in range(N_pwe):
            gx_m, gy_m = G_vectors[m]
            for n in range(N_pwe):
                gx_n, gy_n = G_vectors[n]
                diff_gx = gx_m - gx_n
                diff_gy = gy_m - gy_n
                
                # Diagonal term
                if m == n:
                    M[m, n] = kG_sq[m] * eps_avg_inv + coeff_dict.get((0,0), 1.0)
                else:
                    # Off-diagonal
                    val = coeff_dict.get((diff_gx, diff_gy), 0.0)
                    M[m, n] = val
                    # Note: The diagonal term in the sum is (k+G)^2 / eps(0) + 1/eps(0)
                    # Wait, the formula is sum_G' [ (k+G)^2 delta / eps(0) + 1/eps(G-G') ]
                    # So for m!=n, it is just 1/eps(G_m - G_n).
                    # For m==n, it is (k+G_m)^2 / eps(0) + 1/eps(0).
        
        # Solve eigenvalue problem
        # M is Hermitian if 1/eps is real and symmetric.
        # We can use np.linalg.eigh for Hermitian matrices.
        # However, M might not be perfectly Hermitian due to truncation/numerical issues.
        # But 1/eps is real, so coeffs are conjugate symmetric.
        # M_{mn} = coeff(G_m - G_n). M_{nm} = coeff(G_n - G_m) = coeff(-(G_m - G_n)) = conj(coeff(G_m - G_n)).
        # So M is Hermitian.
        
        try:
            eigenvalues, _ = np.linalg.eigh(M)
            # Sort eigenvalues (eigh returns sorted)
            # We want the 4th and 5th bands (indices 3 and 4)
            # The bands are ordered by omega^2.
            if len(eigenvalues) >= 5:
                return eigenvalues[:5]
            else:
                return np.zeros(5)
        except:
            return np.zeros(5)

    def evaluate_gap(cell):
        """
        Evaluate the gap-midgap ratio for bands 4 and 5.
        """
        # Sample k-points in the irreducible BZ
        # Irreducible BZ for square lattice: 0 <= ky <= kx, kx in [0, pi], ky in [0, pi/2]
        # Actually, the full BZ is [-pi, pi] x [-pi, pi].
        # Irreducible wedge: 0 <= ky <= kx, 0 <= kx <= pi.
        
        # We use a small grid of k-points to estimate the band structure.
        # 5x5 grid in the wedge should be enough for a local search.
        k_points = []
        N_k = 5
        for i in range(N_k + 1):
            kx = math.pi * i / N_k
            for j in range(N_k + 1):
                ky = kx * j / N_k
                k_points.append((kx, ky))
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            eigs = solve_bands(cell, kx, ky)
            # eigs are omega^2
            # omega = sqrt(eig)
            omegas = np.sqrt(eigs)
            max_omega_4 = max(max_omega_4, omegas[3])
            min_omega_5 = min(min_omega_5, omegas[4])
            
        if min_omega_5 < max_omega_4:
            gap_ratio = -1.0 # Overlap
        else:
            midgap = (min_omega_5 + max_omega_4) / 2.0
            if midgap == 0:
                gap_ratio = 0.0
            else:
                gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / midgap
                
        return gap_ratio

    # 2. Initial Design
    # Start with "Holes in Dielectric" (inverse of rods)
    # Background is dielectric (1), holes are air (0).
    # 4 holes in quadrants.
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Create 4 circular holes
    r_hole = 6.0
    centers = [
        (size/4, size/4),
        (size/4, 3*size/4),
        (3*size/4, size/4),
        (3*size/4, 3*size/4)
    ]
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx, cy in centers:
                if (x - cx)**2 + (y - cy)**2 < r_hole**2:
                    cell[i][j] = 0
                    break

    # Ensure symmetry (it should be, but let's enforce)
    def enforce_symmetry(c):
        # c is a list of lists
        # Symmetry: c[i][j] == c[j][i] == c[size-1-i][size-1-j] etc.
        # We define the top-left quadrant (0..15, 0..15) and mirror.
        # Actually, the problem says: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
        # This implies 4-fold symmetry + inversion.
        # The fundamental domain is the top-left 16x16 quadrant?
        # If we set pixels in [0:16, 0:16], we can derive the rest.
        # (i, j) -> (j, i) [transpose]
        # (i, j) -> (size-1-i, size-1-j) [inversion]
        # (i, j) -> (size-1-j, size-1-i) [transpose + inversion]
        
        new_c = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size // 2):
            for j in range(size // 2):
                val = c[i][j]
                # 1. Top-Left
                new_c[i][j] = val
                # 2. Top-Right (mirror x) -> (i, size-1-j)
                new_c[i][size-1-j] = val
                # 3. Bottom-Left (mirror y) -> (size-1-i, j)
                new_c[size-1-i][j] = val
                # 4. Bottom-Right (mirror both) -> (size-1-i, size-1-j)
                new_c[size-1-i][size-1-j] = val
                
                # Also transpose symmetry?
                # The problem says: cell == its transpose.
                # So c[i][j] must equal c[j][i].
                # If we just mirror, does it guarantee transpose?
                # c[i][j] = val. c[j][i] must be val.
                # In my mapping:
                # If i < j, c[j][i] is in the top-right or bottom-left?
                # Let's check:
                # If we set c[i][j] for i,j in 0..15.
                # c[j][i] is also in 0..15.
                # So we must enforce c[i][j] == c[j][i] within the quadrant too.
                # This means the quadrant itself must be symmetric about the diagonal i=j.
                
                # So, we only iterate i <= j in the quadrant?
                # Or we average them?
                # Let's just enforce:
                # For i,j in 0..15:
                #   val = c[i][j]
                #   c[j][i] must be val.
                #   So we should only pick the "upper triangle" of the quadrant as free variables.
                
        # Let's rewrite enforce_symmetry to be robust.
        # Free variables: i in 0..15, j in i..15.
        # Total free vars: 16*17/2 = 136.
        
        new_c = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size // 2):
            for j in range(i, size // 2):
                val = c[i][j]
                # 1. (i, j)
                new_c[i][j] = val
                # 2. (j, i) [Transpose]
                new_c[j][i] = val
                # 3. (size-1-i, size-1-j) [Inversion]
                new_c[size-1-i][size-1-j] = val
                # 4. (size-1-j, size-1-i) [Transpose + Inversion]
                new_c[size-1-j][size-1-i] = val
                
                # Check if any of these overlap in the fundamental domain?
                # If i=j, (i,i) and (j,i) are same.
                # If i != j, they are distinct.
                # Are (i,j) and (size-1-i, size-1-j) distinct?
                # Yes, since i < 16, size-1-i >= 16.
                
        return new_c

    cell = enforce_symmetry(cell)
    
    current_score = evaluate_gap(cell)
    best_cell = [row[:] for row in cell]
    best_score = current_score
    
    # 3. Optimization Loop
    # Hill Climbing: Flip pixels in the fundamental domain and check if score improves.
    
    # Get the list of free pixel indices (i, j) in the fundamental domain
    free_pixels = [(i, j) for i in range(size // 2) for j in range(i, size // 2)]
    
    # Shuffle to avoid bias
    np.random.seed(42)
    indices = np.arange(len(free_pixels))
    np.random.shuffle(indices)
    
    max_iters = 1000
    iter_count = 0
    
    while time.time() < deadline:
        improved = False
        # Try a batch of random flips
        for idx in indices:
            if time.time() > deadline - 10: break
            i, j = free_pixels[idx]
            
            # Flip the pixel
            old_val = cell[i][j]
            new_val = 1 - old_val
            
            # Apply flip to all symmetric positions
            positions = [
                (i, j),
                (j, i),
                (size-1-i, size-1-j),
                (size-1-j, size-1-i)
            ]
            
            # Check for duplicates in positions (e.g. if i==j)
            unique_positions = list(set(positions))
            
            for pi, pj in unique_positions:
                cell[pi][pj] = new_val
                
            # Evaluate
            new_score = evaluate_gap(cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                # Revert
                for pi, pj in unique_positions:
                    cell[pi][pj] = old_val
            
            iter_count += 1
            
        if not improved:
            # If no improvement in a full pass, try a different strategy or stop
            # To avoid getting stuck, we can randomize the order again
            np.random.shuffle(indices)
            # Or break if we've tried many times
            if iter_count > max_iters:
                break

    # Return the best cell found
    return best_cell
