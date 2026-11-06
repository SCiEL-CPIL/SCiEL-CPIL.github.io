import time
import numpy as np
import sys

def evaluate(cell, k_points=None):
    """
    Evaluate the gap-midgap ratio for the 4th and 5th TM bands.
    cell: 32x32 list of lists of 0/1.
    """
    size = 32
    eps_low = 1.0
    eps_high = 12.0
    
    # Convert to numpy array
    cell_np = np.array(cell, dtype=np.float64)
    
    # Calculate 1/epsilon
    # 1 -> 1/12, 0 -> 1/1
    inv_eps = np.where(cell_np == 1, 1.0/eps_high, 1.0/eps_low)
    
    # Compute Fourier coefficients of 1/epsilon
    # We need G vectors for 289 plane waves.
    # Assuming a square grid of G vectors: Gx, Gy in [-8, 8] * 2*pi
    # Let's use units where lattice constant a=1, so G = 2*pi*(nx, ny)
    # We work in units of 2*pi. So nx, ny in [-8, 8].
    
    # FFT of the pixel map
    # The pixel map is on a 32x32 grid.
    # The Fourier coefficient for G=(nx, ny) is (1/N^2) * sum_{i,j} f(i,j) exp(-i 2*pi (nx i + ny j) / N)
    # This is exactly the 2D FFT scaled by 1/N^2.
    
    fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
    
    # Map of G vectors to indices in the FFT array
    # FFT array indices: 0..31.
    # Frequency index k corresponds to k * 2*pi / N.
    # We want G = nx * 2*pi.
    # So frequency index = nx * N / N = nx? No.
    # FFT index m corresponds to frequency m * (2*pi/N) for m=0..N/2, and (m-N)*(2*pi/N) for m>N/2.
    # We want G_x = nx * 2*pi.
    # So m * (2*pi/32) = nx * 2*pi => m = 32 * nx.
    # Since nx is integer, m is a multiple of 32.
    # But our FFT grid is 32x32, so frequencies are 0, 1, ..., 15, -16, ..., -1. (in units of 2*pi/32).
    # This means we can only resolve G vectors that are multiples of 1 in these units?
    # Wait. The lattice constant is 1. The unit cell is 1x1.
    # G vectors are 2*pi * (integer, integer).
    # The discrete Fourier series of the function on the unit cell:
    # f(r) = sum_G f_G exp(i G . r)
    # f_G = integral f(r) exp(-i G . r) d^2r
    # Approximated by sum_{i,j} f(i,j) exp(-i G . (i,j)/N) / N^2.
    # If G = 2*pi * (nx, ny), then exp(-i 2*pi (nx i + ny j) / N).
    # This matches the FFT definition if we identify the FFT bin index m with nx.
    # But the FFT bin index m goes from 0 to N-1.
    # nx can be negative.
    # Bin m corresponds to nx = m if m <= N/2, and nx = m - N if m > N/2.
    # So we can extract f_G for any integer nx, ny by looking up the correct bin.
    
    def get_coeff(nx, ny):
        # Map nx, ny to FFT bin indices
        # nx is in [-8, 8]
        # If nx >= 0: bin = nx
        # If nx < 0: bin = nx + 32
        bin_x = nx if nx >= 0 else nx + size
        bin_y = ny if ny >= 0 else ny + size
        return fft_inv_eps[bin_x, bin_y]
        
    # Define G vectors for 289 plane waves
    # nx, ny in [-8, 8]
    g_vectors = [(nx, ny) for nx in range(-8, 9) for ny in range(-8, 9)]
    N_g = len(g_vectors)
    
    # Precompute matrix structure parts that don't depend on k
    # M_{ij} = (K+G_i)^2 * inv_eps_{G_i - G_j}
    # Note: The standard eigenvalue equation for TM is:
    # sum_j (K+G_i) * (K+G_j) * (1/epsilon)_{G_i-G_j} * A_j = omega^2 A_i
    # Wait, the term is (K+G_i) dot (K+G_j) ?
    # For scalar wave equation in 2D:
    # - (1/epsilon) nabla^2 E_z = omega^2 E_z
    # Expand E_z = sum A_G exp(i (K+G) . r)
    # nabla^2 E_z = - sum A_G (K+G)^2 exp(i (K+G) . r)
    # So: sum_G' (1/epsilon)_{G-G'} A_{G'} (K+G)^2 = omega^2 A_G
    # Yes.
    
    # Precompute inv_eps differences
    # diff_coeffs[i, j] = (1/epsilon)_{G_i - G_j}
    diff_coeffs = np.zeros((N_g, N_g), dtype=np.complex128)
    for i, (nx_i, ny_i) in enumerate(g_vectors):
        for j, (nx_j, ny_j) in enumerate(g_vectors):
            nx_diff = nx_i - nx_j
            ny_diff = ny_i - ny_j
            diff_coeffs[i, j] = get_coeff(nx_diff, ny_diff)
            
    # Define k points in the irreducible Brillouin zone
    # Irreducible BZ is the triangle (0,0) to (pi,0) to (pi,pi).
    # In units of 2*pi, kx, ky in [0, 0.5].
    # Triangle: 0 <= ky <= kx <= 0.5.
    
    if k_points is None:
        # Generate 30 points covering the triangle
        k_points = []
        # Points on the boundary and interior
        # We can just sample a grid and filter
        n_k = 6
        for i in range(n_k + 1):
            for j in range(n_k + 1):
                kx = i / (2 * n_k) # 0 to 0.5
                ky = j / (2 * n_k) # 0 to 0.5
                if ky <= kx:
                    k_points.append((kx, ky))
        # Ensure we have at least 30, or just use what we have. 
        # The evaluator uses 30 specific points. We'll use a dense enough set.
        
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        # Construct matrix
        # M_{ij} = (K+G_i) dot (K+G_j) * diff_coeffs[i, j]
        # Wait, the term is (K+G_i) * (K+G_j) ?
        # The Laplacian gives (K+G_i)^2.
        # The convolution with 1/epsilon mixes G.
        # Equation: sum_j (K+G_i)^2 * (1/epsilon)_{G_i-G_j} * A_j = omega^2 A_i
        # So M_{ij} = (K+G_i)^2 * diff_coeffs[i, j]
        
        k_vec = np.array([kx * 2 * np.pi, ky * 2 * np.pi])
        
        # Calculate (K+G_i)^2 for each G_i
        # G_i is 2*pi * (nx_i, ny_i)
        # K is 2*pi * (kx, ky)
        # K+G_i = 2*pi * (kx+nx_i, ky+ny_i)
        # (K+G_i)^2 = 4*pi^2 * ((kx+nx_i)^2 + (ky+ny_i)^2)
        
        g_vecs = np.array(g_vectors) # shape (N_g, 2)
        k_plus_g = k_vec[None, :] + 2 * np.pi * g_vecs
        k_sq = np.sum(k_plus_g ** 2, axis=1)
        
        # M = k_sq[:, None] * diff_coeffs
        M = k_sq[:, None] * diff_coeffs
        
        # Solve eigenvalue problem
        # We only need the first 5 eigenvalues
        # M is Hermitian?
        # diff_coeffs[i, j] = (1/epsilon)_{G_i-G_j}
        # (1/epsilon)_{-G} = conj((1/epsilon)_G)
        # So diff_coeffs[j, i] = conj(diff_coeffs[i, j])
        # M_{ji} = (K+G_j)^2 * conj(diff_coeffs[i, j])
        # M_{ij}^* = (K+G_i)^2 * conj(diff_coeffs[i, j])
        # M is NOT Hermitian unless k_sq[i] == k_sq[j].
        # Wait. The eigenvalue problem for photonic crystals is usually Hermitian if formulated correctly.
        # Let's check the standard formulation.
        # The matrix is often written as:
        # H_{ij} = (K+G_i) dot (K+G_j) * (1/epsilon)_{G_i-G_j}
        # This H is Hermitian.
        # My previous formula had (K+G_i)^2.
        # Let's re-derive.
        # - (1/epsilon) nabla^2 E = omega^2 E
        # E = sum A_G exp(i(K+G)r)
        # nabla^2 E = - sum A_G |K+G|^2 exp(i(K+G)r)
        # - (1/epsilon) * (- sum A_G |K+G|^2 exp(i(K+G)r)) = omega^2 E
        # sum_G' (1/epsilon)_{G-G'} A_{G'} |K+G'|^2 exp(i(K+G)r) = omega^2 A_G
        # So the matrix element is M_{G, G'} = (1/epsilon)_{G-G'} |K+G'|^2
        # This matrix is NOT Hermitian.
        # However, we can transform it to a Hermitian form by multiplying by sqrt(epsilon) or similar?
        # Or simply, the eigenvalues are real.
        # NumPy's eigh requires Hermitian.
        # We can use eig for non-Hermitian, but it's slower and less stable.
        # Let's check if the matrix is similar to a Hermitian one.
        # Let D = diag(1/sqrt(epsilon_G))? No, epsilon_G is not the diagonal.
        
        # Actually, many codes use the form:
        # sum_j (K+G_i) dot (K+G_j) (1/epsilon)_{G_i-G_j} A_j = omega^2 A_i
        # Let's check if this is the correct equation.
        # If we assume E is expanded in a basis of plane waves, the inner product involves 1/epsilon?
        # The standard "Plane Wave Method" (Joannopoulos) uses:
        # sum_j (K+G_i) dot (K+G_j) (1/epsilon)_{G_i-G_j} A_j = omega^2 A_i
        # This matrix IS Hermitian.
        # Let's verify:
        # M_{ji} = (K+G_j) dot (K+G_i) (1/epsilon)_{G_j-G_i}
        # M_{ij}^* = (K+G_i) dot (K+G_j) (1/epsilon)_{G_i-G_j}^*
        # Since (1/epsilon)_{-G} = (1/epsilon)_G^*, we have (1/epsilon)_{G_j-G_i} = (1/epsilon)_{-(G_i-G_j)} = (1/epsilon)_{G_i-G_j}^*.
        # And dot product is real and symmetric.
        # So M_{ji} = M_{ij}^*.
        # Yes, this matrix is Hermitian.
        
        # So we should use M_{ij} = (K+G_i) dot (K+G_j) * diff_coeffs[i, j]
        
        k_dot_kg = np.sum(k_plus_g * k_plus_g.T, axis=2) # This is (K+Gi) dot (K+Gj)
        # Wait, k_plus_g is shape (N_g, 2).
        # k_dot_kg[i, j] = k_plus_g[i] dot k_plus_g[j]
        
        M = k_dot_kg * diff_coeffs
        
        # Make sure it's exactly Hermitian
        M = (M + M.conj().T) / 2.0
        
        # Compute eigenvalues
        # We only need the first 5
        # np.linalg.eigvalsh is efficient
        eigvals = np.linalg.eigvalsh(M)
        
        # The first 5 eigenvalues correspond to bands 1, 2, 3, 4, 5
        # Note: omega^2 are the eigenvalues.
        # We want omega.
        # omega_4 = sqrt(eigvals[3])
        # omega_5 = sqrt(eigvals[4])
        
        w4 = np.sqrt(eigvals[3])
        w5 = np.sqrt(eigvals[4])
        
        if w4 > max_omega_4:
            max_omega_4 = w4
        if w5 < min_omega_5:
            min_omega_5 = w5
            
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    
    if midgap == 0:
        return 0.0
        
    ratio = gap / midgap
    return ratio

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial design: Square dielectric
    # 1/8th domain: 0 <= j <= i < 16
    # Let's start with a square of size 12x12 in the corner?
    # Or centered?
    # The seed was a circle.
    # Let's try a square centered in the cell.
    # Cell is 32x32.
    # Square from 8 to 23? (16x16).
    # Or 10 to 21? (12x12).
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    # Initialize with a square dielectric region
    # Let's try a square of half-width 0.3 (approx 10 pixels)
    half_w = 10
    for i in range(size):
        for j in range(size):
            # Center is 15.5
            # Distance from center
            # We want a square centered at 15.5
            # |i - 15.5| <= half_w and |j - 15.5| <= half_w
            if abs(i - 15.5) <= half_w and abs(j - 15.5) <= half_w:
                cell[i][j] = 1
                
    best_score = evaluate(cell)
    best_cell = [row[:] for row in cell]
    
    # Local search
    # Fundamental domain: 0 <= j <= i < 16
    # Pixels to consider:
    fund_pixels = []
    for i in range(16):
        for j in range(i + 1):
            fund_pixels.append((i, j))
            
    np.random.seed(42)
    
    while time.time() < deadline:
        # Pick a random pixel in the fundamental domain
        idx = np.random.randint(len(fund_pixels))
        i, j = fund_pixels[idx]
        
        # Flip it
        cell[i][j] = 1 - cell[i][j]
        
        # Propagate symmetry
        # 1. Transpose
        # cell[i][j] changed.
        # cell[j][i] must be same.
        if i != j:
            cell[j][i] = cell[i][j]
            
        # 2. 180 deg rotation (center symmetry)
        # cell[31-i][31-j] must be same.
        i2 = 31 - i
        j2 = 31 - j
        if (i, j) != (i2, j2) and (i, j) != (j, i) and (i2, j2) != (j2, i2):
             # We need to update all 4 symmetric positions if they are distinct
             # Actually, just setting the 4 positions to the value is safest
             pass
             
        # Let's just explicitly set the 4 symmetric positions to ensure consistency
        val = cell[i][j]
        cell[i][j] = val
        if j != i: cell[j][i] = val
        cell[31-i][31-j] = val
        if j != i: cell[31-j][31-i] = val
        
        score = evaluate(cell)
        
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
        else:
            # Revert
            cell[i][j] = 1 - val
            if j != i: cell[j][i] = 1 - val
            cell[31-i][31-j] = 1 - val
            if j != i: cell[31-j][31-i] = 1 - val
            
    return best_cell

