import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    delta_eps = eps_high - eps_low
    
    # The fundamental domain is the upper triangle of the 16x16 top-left block.
    # Total 136 pixels.
    # We will represent the state as a 16x16 array, but only modify the upper triangle.
    # Actually, it's easier to just keep the 136 values in a list or a 1D array.
    # Let's use a 16x16 array and enforce symmetry in the expansion step.
    
    # Initialize with Dielectric background (1) and central Air hole (0)
    # Center of 32x32 is (15.5, 15.5).
    # In 16x16 block (0..15), center is bottom-right corner (15.5, 15.5).
    
    base_16 = np.ones((16, 16), dtype=int)
    
    # Create a hole. Radius R.
    # Let's start with R=5.
    R = 5.0
    for i in range(16):
        for j in range(16):
            # Pixel center in 32x32 coords: (i+0.5, j+0.5)
            # Distance to center (15.5, 15.5)
            dx = 15.5 - (i + 0.5)
            dy = 15.5 - (j + 0.5)
            if dx*dx + dy*dy < R*R:
                base_16[i][j] = 0
                
    # Enforce transpose symmetry in the 16x16 block for initialization
    # The 16x16 block must be symmetric because the full cell is.
    # C[i][j] = C[j][i] for i,j in 0..15.
    # So we average or just copy upper to lower.
    # Since it's binary, if there's a conflict, we might need to resolve.
    # For a central hole, it should be symmetric anyway.
    for i in range(16):
        for j in range(i):
            if base_16[i][j] != base_16[j][i]:
                # Conflict: pick one? Or average?
                # In a valid design, they should be same.
                # For init, we just force symmetry.
                base_16[i][j] = base_16[j][i]
                
    best_16 = base_16.copy()
    
    # Surrogate parameters
    N_pw = 13 # 13x13 = 169 waves
    # G vectors
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vecs = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_waves = len(G_vecs)
    
    # Precompute G-G' differences indices for epsilon convolution
    # We need epsilon(G-G') for all pairs G, G'
    # G indices are 0..N_waves-1
    # We can compute epsilon_G for all G in the basis
    # Then H[G, G'] = epsilon(G-G') * |k+G'|^2
    
    # K points in irreducible BZ
    # Square BZ: -pi/a to pi/a.
    # Irreducible wedge: 0 <= kx <= ky <= pi/a?
    # Or 0 <= kx <= pi/2, 0 <= ky <= pi/2, kx <= ky?
    # Standard irreducible zone for square: 0 <= kx <= pi/2, 0 <= ky <= pi/2, kx <= ky.
    # Points:
    # Gamma (0,0)
    # X (pi, 0) -> (1, 0) in units of pi/a? No, BZ is -pi to pi.
    # Let's use units where lattice constant a=1. BZ is [-pi, pi].
    # Irreducible: 0 <= kx <= pi/2, 0 <= ky <= pi/2, kx <= ky.
    # Key points:
    # Gamma: (0,0)
    # M: (pi/2, pi/2)
    # X: (pi, 0) is not in this wedge?
    # The boundary of the BZ is at kx=pi/2 or ky=pi/2?
    # For square lattice, reciprocal lattice vectors are (2pi, 0), (0, 2pi).
    # BZ is [-pi, pi] x [-pi, pi].
    # High symmetry points:
    # Gamma (0,0)
    # X (pi, 0)
    # M (pi, pi)
    # Irreducible wedge: 0 <= kx <= pi/2, 0 <= ky <= pi/2, kx <= ky.
    # This wedge covers 1/8 of the BZ.
    # Points to sample:
    k_points = [
        (0.0, 0.0),      # Gamma
        (np.pi/2, 0.0),  # This is on the boundary? No, X is (pi, 0).
        # Wait, the BZ boundary for square lattice is at kx = +/- pi/2?
        # No, the first BZ is defined by Bragg planes.
        # For square lattice a=1, G=(2pi, 0). Bragg plane at kx = pi.
        # So BZ is [-pi, pi].
        # High symmetry: Gamma(0,0), X(pi, 0), M(pi, pi).
        # Irreducible: 0 <= kx <= pi/2, 0 <= ky <= pi/2, kx <= ky.
        # This triangle has vertices (0,0), (pi/2, pi/2), (pi/2, 0)?
        # No, (pi/2, 0) is not a high symmetry point of the *triangle* but lies on the boundary kx=ky? No.
        # The line kx=ky goes from (0,0) to (pi/2, pi/2) (M point).
        # The line kx=0 goes from (0,0) to (0, pi/2)?
        # The line ky=kx is the symmetry axis.
        # The other boundaries are kx=0 and ky=pi/2?
        # Let's just sample a grid in the triangle 0<=kx<=pi/2, 0<=ky<=pi/2, kx<=ky.
        
        (0.0, 0.0),
        (np.pi/2, np.pi/2), # M
        (np.pi/2, 0.0),     # This is (pi/2, 0). Is this in the wedge? kx=pi/2, ky=0. kx > ky. So no.
        # The wedge is kx <= ky.
        # So (0, pi/2) is in the wedge.
        (0.0, np.pi/2),
        (np.pi/4, np.pi/4),
        (np.pi/8, np.pi/4),
        (np.pi/4, np.pi/2),
    ]
    
    # Remove duplicates and ensure within bounds
    k_points = list(set([(round(k[0], 5), round(k[1], 5)) for k in k_points]))
    
    def expand_to_32(block_16):
        # block_16 is 16x16, symmetric.
        # Full cell 32x32.
        # C[i][j] = block_16[i%16 if i<16 else 15-(i-16)][j%16 if j<16 else 15-(j-16)]?
        # Symmetry: C[i][j] = C[31-i][j] = C[i][31-j].
        # So for i in 0..31, map to i' in 0..15.
        # If i < 16, i' = i.
        # If i >= 16, i' = 31 - i.
        # Same for j.
        # So C[i][j] = block_16[i'][j'].
        cell = np.zeros((32, 32), dtype=int)
        for i in range(32):
            i_prime = i if i < 16 else 31 - i
            for j in range(32):
                j_prime = j if j < 16 else 31 - j
                cell[i][j] = block_16[i_prime][j_prime]
        return cell

    def compute_eps_fourier(cell_32):
        # cell_32 is 32x32 0/1.
        # epsilon = 1 + 11 * mask.
        # We need epsilon(G) for G in the plane wave basis.
        # G vectors are in units of 2pi/32?
        # The pixel grid is 32x32. The lattice constant is 1.
        # The pixel size is 1/32.
        # The Fourier transform of the pixelated function.
        # epsilon(G) = (1/32^2) sum_{i,j} epsilon(i,j) exp(-i G . (i,j))
        # G must be integer multiples of 2pi/32?
        # The plane wave basis G vectors are typically integer multiples of 2pi/a.
        # Here a=1. So G = (2pi * n, 2pi * m).
        # In the sum, the phase is exp(-i 2pi (n i + m j) / 32).
        # So we can use np.fft.fftn on the 32x32 array.
        # The FFT gives coefficients for frequencies 0, 1, ..., 31.
        # These correspond to G = 2pi * k / 32.
        # We need G = 2pi * n.
        # So we need FFT bins n = 0, 1, ..., 15 (and negatives).
        # Bin 0 -> G=0.
        # Bin 1 -> G=2pi/32. This is NOT a reciprocal lattice vector.
        # The reciprocal lattice vectors are G = (2pi, 0), (0, 2pi), etc.
        # So we need FFT bins that are multiples of 32?
        # No, the function is periodic with period 1.
        # The pixel grid has 32 pixels per period.
        # The DFT of the 32x32 array gives coefficients for frequencies k = 0, ..., 31.
        # The physical frequency is 2pi * k / 32.
        # The reciprocal lattice vectors are 2pi * m.
        # So we need k = 32 * m.
        # For m=0, k=0.
        # For m=1, k=32. But FFT size is 32, so k=32 is alias of k=0.
        # This implies that a pixelated function with 32 pixels cannot resolve the fundamental reciprocal lattice vector G=2pi?
        # Wait. The plane wave expansion uses G vectors that are reciprocal lattice vectors.
        # G = h * b1 + k * b2. b1 = (2pi, 0).
        # So G = (2pi h, 2pi k).
        # The Fourier coefficient of the dielectric function at G is:
        # epsilon(G) = integral epsilon(r) e^{-i G r} dr.
        # Approximated by sum_{i,j} epsilon(i/32, j/32) e^{-i G (i/32, j/32)} / 32^2.
        # Phase: exp(-i 2pi h i / 32 * 32 / 32)?
        # G . r = (2pi h) * (i/32) = 2pi h i / 32.
        # So we need the FFT of the 32x32 array at bin h.
        # Yes! The FFT bin h corresponds to G = 2pi h / 32 * 32?
        # No.
        # FFT bin h corresponds to frequency 2pi h / N.
        # Here N=32. So frequency is 2pi h / 32.
        # But we need G = 2pi * integer.
        # So we need h such that 2pi h / 32 = 2pi * m => h = 32 m.
        # Since h is in 0..31, only h=0 is valid?
        # This suggests that the pixelated representation with 32 pixels is too coarse to represent the periodicity?
        # No, the pixelated function IS the function.
        # The "exact Fourier coefficients of the pixel map" mentioned in the prompt.
        # The pixel map is a sum of rectangles.
        # The Fourier coefficient of a rectangle is a sinc function.
        # So we should compute the analytic Fourier coefficients of the pixel map, not the DFT.
        # Or, the evaluator uses the DFT of the pixel values?
        # "exact Fourier coefficients of the pixel map".
        # If the pixel map is treated as a step function (piecewise constant), the Fourier coefficients are:
        # epsilon(G) = (1/32^2) sum_{i,j} val(i,j) * [ (1 - e^{-i G_x / 32}) / (i G_x / 32) ] * [ (1 - e^{-i G_y / 32}) / (i G_y / 32) ] * e^{-i G . (i/32, j/32)}
        # This is the correct analytic FT of the pixelated function.
        
        # For speed in the surrogate, we can approximate this.
        # Or just use the DFT?
        # If the evaluator uses the analytic FT, using DFT in the surrogate might be inaccurate.
        # However, for low G vectors, the DFT is a good approximation if we account for the pixel shape.
        # Let's implement the analytic FT for the surrogate too, to be consistent.
        
        eps_G = np.zeros(N_waves, dtype=complex)
        # G_vecs are (h, k) in units of 2pi/a?
        # My G_vecs above were integers. Let's assume they are h, k.
        # So G_x = 2pi * h, G_y = 2pi * k.
        
        for idx, (h, k) in enumerate(G_vecs):
            if h == 0 and k == 0:
                # DC term
                eps_G[idx] = 1 + delta_eps * np.mean(cell_32)
                continue
            
            # Analytic FT of pixelated function
            # Sum over pixels
            # This is slow if done naively for every G.
            # But N_waves is 169. 32x32 = 1024 pixels.
            # 169 * 1024 = 173k ops. Very fast.
            
            sum_val = 0.0
            for i in range(32):
                for j in range(32):
                    val = cell_32[i][j]
                    if val == 0:
                        continue
                    # Pixel center (i+0.5, j+0.5) / 32
                    # Phase factor
                    phase = -2.0 * np.pi * (h * (i + 0.5) / 32.0 + k * (j + 0.5) / 32.0)
                    # Pixel shape factor (sinc)
                    # Integral of exp(-i G x) over [-0.5, 0.5] in pixel units
                    # = 2 * sin(G/2) / G
                    # G in pixel units: G_pix = 2pi * h / 32 * 32 = 2pi h.
                    # Wait, G is in units of 1/a. Pixel width is 1/32 a.
                    # G_pix = G * (1/32) = 2pi h / 32.
                    # Sinc factor: sin(G_pix/2) / (G_pix/2)
                    sinc_x = 1.0
                    sinc_y = 1.0
                    if h != 0:
                        gx = 2.0 * np.pi * h / 32.0
                        sinc_x = np.sin(gx/2.0) / (gx/2.0)
                    if k != 0:
                        gy = 2.0 * np.pi * k / 32.0
                        sinc_y = np.sin(gy/2.0) / (gy/2.0)
                    
                    sum_val += val * np.exp(1j * phase) * sinc_x * sinc_y
            
            eps_G[idx] = 1 + delta_eps * (sum_val / 1024.0)
            
        return eps_G

    def surrogate_gap(block_16):
        cell_32 = expand_to_32(block_16)
        eps_G = compute_eps_fourier(cell_32)
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build Hamiltonian
            # H[G, G'] = epsilon(G-G') * |k+G'|^2
            # k is in units of 2pi/a.
            # G is in units of 2pi/a.
            
            H = np.zeros((N_waves, N_waves), dtype=complex)
            
            # Precompute |k+G|^2 for all G
            k_plus_G_sq = np.zeros(N_waves)
            for idx, (h, k) in enumerate(G_vecs):
                kx_g = kx + 2.0 * np.pi * h
                ky_g = ky + 2.0 * np.pi * k
                k_plus_G_sq[idx] = kx_g**2 + ky_g**2
            
            # We need epsilon(G-G')
            # G-G' is a vector (h1-h2, k1-k2).
            # We have eps_G for all G in the basis.
            # But G-G' might not be in the basis?
            # The basis is a square grid of G vectors.
            # If we use a square grid, G-G' is also in the grid if the grid is closed under subtraction?
            # Yes, if we use a finite square grid, the difference is within the range.
            # We need a mapping from (h, k) to index.
            
            # Create a dictionary or 2D array for eps_G
            # G_vals is -N//2 to N//2.
            # Offset = N//2.
            offset = N_pw // 2
            eps_grid = np.zeros((N_pw, N_pw), dtype=complex)
            for idx, (h, k) in enumerate(G_vecs):
                # h, k are integers
                # Map to grid index
                # G_vals = [-6, ..., 6] for N=13.
                # Index in G_vals array:
                # Let's just use a dict for simplicity in surrogate.
                pass
            
            # Better: use the grid structure.
            # G_vecs are generated from meshgrid of G_vals.
            # So we can index by position in G_vals.
            # G_vals = np.arange(-6, 7).
            # idx_h = h - G_vals[0]
            
            eps_lookup = np.zeros((len(G_vals), len(G_vals)), dtype=complex)
            for idx, (h, k) in enumerate(G_vecs):
                ih = h - G_vals[0]
                ik = k - G_vals[0]
                eps_lookup[ih, ik] = eps_G[idx]
                
            for i in range(N_waves):
                hi, ki = G_vecs[i]
                ih = hi - G_vals[0]
                ik = ki - G_vals[0]
                for j in range(N_waves):
                    hj, kj = G_vecs[j]
                    dh = hi - hj
                    dk = ki - kj
                    # Get epsilon(dh, dk)
                    # If (dh, dk) is in the grid
                    jh = dh + G_vals[0]
                    jk = dk + G_vals[0]
                    if 0 <= jh < len(G_vals) and 0 <= jk < len(G_vals):
                        eps_val = eps_lookup[jh, jk]
                    else:
                        eps_val = 1.0 # Approximation for high G
                    
                    H[i, j] = eps_val * k_plus_G_sq[j]
                    
            # Solve eigenvalue problem
            # H is Hermitian?
            # epsilon(-G) = conj(epsilon(G)).
            # H[i, j] = eps(G_i - G_j) |k+G_j|^2.
            # H[j, i] = eps(G_j - G_i) |k+G_i|^2.
            # H[i, j] / H[j, i] = eps(G_i-G_j)/eps(G_j-G_i) * |k+G_j|^2/|k+G_i|^2.
            # Not necessarily Hermitian unless we weight the basis?
            # Standard PW method uses a generalized eigenvalue problem or symmetrizes.
            # The matrix A_{GG'} = eps(G-G') (k+G) . (k+G') is NOT symmetric.
            # However, the operator is self-adjoint.
            # We can use np.linalg.eigvalsh on a symmetrized version?
            # Or solve det(H - omega^2 I) = 0.
            # For speed, let's assume we can use eigvalsh on (H + H.T.conj())/2?
            # This is an approximation.
            
            H_sym = (H + H.T.conj()) / 2.0
            eigs = np.linalg.eigvalsh(H_sym)
            eigs = np.sort(eigs)
            
            # Bands are sqrt(eigs)
            # We want band 4 and 5.
            # Indices 3 and 4 (0-based).
            if len(eigs) >= 5:
                w4 = np.sqrt(eigs[3])
                w5 = np.sqrt(eigs[4])
                if w5 < min_omega_5:
                    min_omega_5 = w5
                if w4 > max_omega_4:
                    max_omega_4 = w4
                    
        if min_omega_5 == np.inf:
            return -1.0
            
        gap = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap

    # Local Search
    best_gap = surrogate_gap(best_16)
    
    # Random seed for reproducibility in search
    np.random.seed(42)
    
    while time.time() < deadline:
        # Pick a random pixel in the 136-pixel domain
        # We can just pick a random i, j in 0..15 and flip if i>=j
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        if i < j:
            i, j = j, i
            
        # Flip
        new_16 = best_16.copy()
        new_16[i][j] = 1 - new_16[i][j]
        # Enforce symmetry
        new_16[j][i] = new_16[i][j]
        
        # Evaluate
        gap = surrogate_gap(new_16)
        
        if gap > best_gap:
            best_16 = new_16
            best_gap = gap
            
    # Final result
    final_cell_32 = expand_to_32(best_16)
    
    # Convert to nested list
    return final_cell_32.tolist()
