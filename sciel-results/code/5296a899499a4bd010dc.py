import numpy as np
import time

def design(problem):
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"] # 1
    eps_high = problem["eps_high"] # 12
    
    # 1. Set up Plane Wave Method components
    # Determine G vectors for 289 plane waves
    # We want the 289 lowest magnitude vectors (n, m)
    candidates = []
    max_N = 15 # Safe upper bound
    for n in range(-max_N, max_N + 1):
        for m in range(-max_N, max_N + 1):
            mag_sq = n*n + m*m
            candidates.append((mag_sq, n, m))
    candidates.sort()
    # Take first 289
    G_vectors = [(n, m) for _, n, m in candidates[:289]]
    G_indices = np.array(G_vectors) # shape (289, 2)
    N_pw = len(G_vectors)
    
    # 30 k-vectors in irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    # Let's sample this triangle.
    k_points = []
    # Add high symmetry points and lines
    # Gamma (0,0)
    k_points.append((0, 0))
    # X (pi, 0)
    k_points.append((np.pi, 0))
    # M (pi/2, pi/2)
    k_points.append((np.pi/2, np.pi/2))
    
    # Sample lines
    # Gamma-X: kx in [0, pi], ky=0
    for i in range(1, 6):
        kx = i * np.pi / 6
        k_points.append((kx, 0))
        
    # X-M: kx=pi, ky in [0, pi/2]
    for i in range(1, 5):
        ky = i * np.pi / 10
        k_points.append((np.pi, ky))
        
    # M-Gamma: kx=ky in [0, pi/2]
    for i in range(1, 5):
        k = i * np.pi / 10
        k_points.append((k, k))
        
    # Interior points to reach 30
    # Generate a grid in the triangle and pick remaining
    # Triangle vertices: (0,0), (pi,0), (pi/2, pi/2)
    # Condition: 0 <= kx <= ky <= pi AND kx + ky <= pi? No.
    # The irreducible wedge is defined by kx >= 0, ky >= 0, kx <= ky, and kx + ky <= pi?
    # Wait, the BZ is a square [-pi, pi] x [-pi, pi].
    # Symmetry group D4.
    # Irreducible sector: 0 <= kx <= ky <= pi.
    # Is that right?
    # Reflection x->-x, y->-y, swap x,y.
    # Yes, 0 <= kx <= ky <= pi is the standard irreducible BZ for square lattice.
    
    # We have 1 + 1 + 1 + 5 + 4 + 4 = 16 points.
    # Need 14 more.
    # Add interior grid points
    step = np.pi / 4
    for i in range(1, 4):
        for j in range(i, 4):
            kx = i * step
            ky = j * step
            if 0 <= kx <= ky <= np.pi:
                # Check if not already added
                if not any(abs(kx - kp[0]) < 1e-6 and abs(ky - kp[1]) < 1e-6 for kp in k_points):
                    k_points.append((kx, ky))
                    
    # If still less than 30, add more
    while len(k_points) < 30:
        # Random or systematic addition
        # Let's just add more grid points
        added = False
        for i in range(1, 8):
            for j in range(i, 8):
                kx = i * np.pi / 12
                ky = j * np.pi / 12
                if 0 <= kx <= ky <= np.pi:
                    if not any(abs(kx - kp[0]) < 1e-6 and abs(ky - kp[1]) < 1e-6 for kp in k_points):
                        k_points.append((kx, ky))
                        added = True
                        break
            if added: break
        if not added:
            break
            
    k_points = np.array(k_points[:30])
    
    def compute_score(cell):
        # cell is 32x32 list of 0/1
        # 1 = Dielectric (eps=12), 0 = Air (eps=1)
        # We need epsilon_inv
        # eps_inv = 1/eps
        # Grid of eps_inv
        eps_inv_grid = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    eps_inv_grid[i][j] = 1.0 / eps_high
                else:
                    eps_inv_grid[i][j] = 1.0 / eps_low
        
        # Compute Fourier coefficients of eps_inv
        # V_GG' = 1/A * sum_{x,y} eps_inv(x,y) exp(-i (G-G') . (x,y))
        # Using FFT
        # The grid is 32x32. The unit cell is 1x1.
        # dx = 1/32.
        # The discrete sum approximates the integral.
        # We can use numpy.fft.fft2
        # Note: numpy.fft.fft2 uses exp(-2pi i (kx x + ky y))
        # Our G vectors are in units of 2pi?
        # Usually G = 2pi * (n, m) / a. Here a=1, so G = 2pi (n,m).
        # The FFT basis functions are exp(2pi i (n x / N + m y / N)).
        # With x = i/N, y = j/N.
        # exp(2pi i (n i / N^2 ...)) No.
        
        # Let's stick to the direct definition or careful FFT mapping.
        # The integral is over x,y in [0,1].
        # V_nm = sum_{i,j} eps_inv[i,j] * exp(-i 2pi (n i/32 + m j/32)) * (1/32^2)
        # This is exactly (1/1024) * FFT2(eps_inv_grid)_{n,m}
        
        # However, we need V for G-G'.
        # G = 2pi (n,m).
        # The FFT gives coefficients for frequencies n, m.
        # So V_{(n,m), (n',m')} = C_{(n-n', m-m')}
        # where C is the DFT of the grid.
        
        # Compute DFT
        # numpy.fft.fft2 returns unnormalized sum.
        # We need the integral, so divide by N^2 = 1024.
        fourier_coeffs = np.fft.fft2(eps_inv_grid) / (size * size)
        
        # We need to access fourier_coeffs[G-G']
        # G vectors are indices in G_vectors.
        # Let's create a lookup for fourier_coeffs by (n,m)
        # But fourier_coeffs is an array of size 32x32.
        # Indices n, m in DFT correspond to frequencies.
        # n in [0, 31]. If n > 16, it's negative frequency n-32.
        
        def get_coeff(n, m):
            # Map n, m to FFT indices
            # FFT index i corresponds to frequency i if i <= N/2, else i-N
            idx_n = n % size
            idx_m = m % size
            return fourier_coeffs[idx_n, idx_m]
            
        # Build matrices for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for kx, ky in k_points:
            # k is in radians?
            # G is 2pi(n,m).
            # The eigenvalue equation: sum_{G'} V_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
            # Wait, for TM, it's det( V * |k+G|^2 I - omega^2 ) = 0?
            # Equation: sum_{G'} V_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
            # Matrix M_{G, G'} = V_{G-G'} * |k+G'|^2
            # We need to solve det(M - omega^2 I) = 0.
            
            M = np.zeros((N_pw, N_pw), dtype=complex)
            for i, (nx, mx) in enumerate(G_vectors):
                # k + G'
                # G' is 2pi (n', m')
                # k is (kx, ky)
                # |k + G'|^2 = (kx + 2pi n')^2 + (ky + 2pi m')^2
                for j, (nxp, mxp) in enumerate(G_vectors):
                    # G - G'
                    dn = nx - nxp
                    dm = mx - mxp
                    V_val = get_coeff(dn, dm)
                    
                    # |k + G'|^2
                    kpx = kx + 2 * np.pi * nxp
                    kpy = ky + 2 * np.pi * mxp
                    k2 = kpx*kpx + kpy*kpy
                    
                    M[i, j] = V_val * k2
            
            # Solve eigenvalues
            # M is Hermitian?
            # V_{G-G'} is Hermitian if eps is real.
            # V_{G-G'} = V_{(G-G')*}^*
            # Since eps is real, V_{-G} = V_G^*.
            # M_{ij} = V_{G_i - G_j} |k+G_j|^2
            # M_{ji} = V_{G_j - G_i} |k+G_i|^2
            # These are not necessarily conjugates of each other unless |k+G_i|^2 = |k+G_j|^2, which is false.
            # So M is not Hermitian.
            # We must use general eigenvalues or reformulate.
            
            # Standard PWE formulation:
            # det( sum_{G'} V_{G-G'} |k+G'|^2 delta_{GG'} - omega^2 delta_{GG'} ) = 0
            # Wait, the equation is:
            # sum_{G'} V_{G-G'} |k+G'|^2 c_{G'} = omega^2 c_G
            # This is a generalized eigenvalue problem? No, it's a standard one if we define the matrix correctly.
            # But the matrix A_{GG'} = V_{G-G'} |k+G'|^2 is not symmetric.
            # However, the operator is self-adjoint in the appropriate inner product.
            # We can use numpy.linalg.eigvals for non-Hermitian, but it's slower and less stable.
            # Alternative: Transform to Hermitian form?
            # Or just use scipy.linalg.eigvals.
            
            # Let's use scipy.linalg.eigvals
            from scipy.linalg import eigvals
            eigs = eigvals(M)
            # We want real positive eigenvalues (omega^2)
            # Filter real parts close to imaginary 0
            real_eigs = eigs.real
            # Sort
            real_eigs.sort()
            
            # Bands 4 and 5 are indices 3 and 4 (0-indexed)
            # But we need to be careful with degeneracies.
            # The problem says "between the fourth and fifth TM bands".
            # So min(omega_5) and max(omega_4).
            # omega_4 is the 4th lowest band. omega_5 is the 5th.
            
            # Take the 5th smallest eigenvalue (index 4) and 4th smallest (index 3)
            if len(real_eigs) >= 5:
                w4_sq = real_eigs[3]
                w5_sq = real_eigs[4]
                w4 = np.sqrt(w4_sq) if w4_sq > 0 else 0
                w5 = np.sqrt(w5_sq) if w5_sq > 0 else 0
                
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    # 2. Search Algorithm
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = -1.0
    best_cell = None
    
    # Generate initial candidate: Inverted background (All 1s) with a central Air hole
    # Shape: Square hole
    def make_square_hole(side):
        cell = [[1 for _ in range(size)] for _ in range(size)]
        # Center is (16, 16)
        # Side length s. Range [16 - s/2, 16 + s/2)
        # Pixel indices i, j in [0, 31]
        # Center pixel is 15.5?
        # Let's use coordinate system where pixel (i,j) covers [i/32, (i+1)/32]
        # Center is 0.5.
        # Hole from 0.5 - side/2 to 0.5 + side/2
        c = 0.5
        half = side / 2.0
        for i in range(size):
            x = (i + 0.5) / size
            if c - half <= x < c + half:
                for j in range(size):
                    y = (j + 0.5) / size
                    if c - half <= y < c + half:
                        cell[i][j] = 0
        return cell

    def make_circle_hole(radius):
        cell = [[1 for _ in range(size)] for _ in range(size)]
        c = 0.5
        for i in range(size):
            x = (i + 0.5) / size - c
            for j in range(size):
                y = (j + 0.5) / size - c
                if x*x + y*y < radius*radius:
                    cell[i][j] = 0
        return cell

    def make_dumbbell_hole(radius, dist):
        cell = [[1 for _ in range(size)] for _ in range(size)]
        c = 0.5
        # Centers at (c +/- dist, c) and (c, c +/- dist)
        centers = [
            (c + dist, c), (c - dist, c),
            (c, c + dist), (c, c - dist)
        ]
        for i in range(size):
            x = (i + 0.5) / size
            for j in range(size):
                y = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        cell[i][j] = 0
                        break
        return cell

    # Grid search parameters
    # Square side: 0.2 to 0.6
    # Circle radius: 0.1 to 0.4
    # Dumbbell radius: 0.1 to 0.3, dist: 0.1 to 0.3
    
    candidates = []
    
    # Add square holes
    for s in np.arange(0.2, 0.7, 0.05):
        candidates.append(('square', s))
        
    # Add circle holes
    for r in np.arange(0.1, 0.5, 0.05):
        candidates.append(('circle', r))
        
    # Add dumbbell holes
    for r in np.arange(0.1, 0.3, 0.05):
        for d in np.arange(0.1, 0.3, 0.05):
            candidates.append(('dumbbell', r, d))

    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
            
        if cand[0] == 'square':
            cell = make_square_hole(cand[1])
        elif cand[0] == 'circle':
            cell = make_circle_hole(cand[1])
        elif cand[0] == 'dumbbell':
            cell = make_dumbbell_hole(cand[1], cand[2])
            
        score = compute_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # If we have time, do a local refinement around the best candidate
    # But for now, return the best found
    
    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell
