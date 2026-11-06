import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Optimizes the radius of 4 dielectric rods in an air background to maximize
    the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Pre-compute coordinates
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    # Define the 2x2 rod positions
    dx = 0.25
    dy = 0.25
    
    # Candidate radii to test
    candidate_radii = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    
    best_gap = -1.0
    best_cell = None
    
    # Plane wave parameters
    # The evaluator uses 289 plane waves. 
    # 289 = 17^2. This implies a cutoff radius of 16 in reciprocal lattice units (since indices -16 to 16).
    # Let's use the same cutoff for consistency.
    cutoff = 16
    # Generate reciprocal lattice vectors
    Gx = np.arange(-cutoff, cutoff + 1)
    Gy = np.arange(-cutoff, cutoff + 1)
    Gx, Gy = np.meshgrid(Gx, Gy)
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Fourier coefficients of epsilon(r)
    # epsilon(r) = eps_low + (eps_high - eps_low) * mask(r)
    # We need epsilon_G = integral epsilon(r) exp(-i G.r) d^2r
    # Since we have a pixel map, we can approximate the integral by summing over pixels.
    # For a pixel at (x_i, y_i) with value v_i (0 or 1), the contribution is v_i * exp(-i G.x_i) * exp(-i G.y_i) / (size*size)
    # Actually, the standard plane wave method uses:
    # epsilon_G = (1/A) ∫_cell epsilon(r) e^{-i G·r} d^2r
    # With pixel discretization:
    # epsilon_G ≈ (1/size^2) * sum_{i,j} mask_{i,j} * exp(-i (Gx_i * x_i + Gy_j * y_j))
    
    # Pre-compute the exponential terms for all G and all pixels?
    # N_pw * size * size = 289 * 1024 = ~296k complex numbers. This is small.
    # Let's pre-compute exp(-i G·r) for all G and all pixel centers.
    
    # Pixel centers
    px_x = x
    px_y = y
    px_X, px_Y = np.meshgrid(px_x, px_y)
    px_X = px_X.ravel()
    px_Y = px_Y.ravel()
    
    # G vectors are in units of 2π/a. Since a=1, G = 2π * (nx, ny).
    # The dot product G·r = 2π * (nx * x + ny * y)
    
    # Compute phase matrix: (N_pw, size*size)
    # G_vectors[:, 0] is nx, G_vectors[:, 1] is ny
    # phases = exp(-1j * 2 * pi * (nx[:, None] * px_X[None, :] + ny[:, None] * px_Y[None, :]))
    nx = G_vectors[:, 0][:, None]
    ny = G_vectors[:, 1][:, None]
    px_X_row = px_X[None, :]
    px_Y_row = px_Y[None, :]
    
    phases = np.exp(-1j * 2 * np.pi * (nx * px_X_row + ny * px_Y_row))
    
    # We need to compute epsilon_G for each candidate cell.
    # epsilon_G = eps_low * delta_{G,0} + (eps_high - eps_low) * (1/size^2) * sum(mask * phases)
    # Note: The mask is binary.
    
    # k-points in the irreducible Brillouin zone
    # The evaluator uses 30 k-points. We will use a representative set covering the IBZ.
    # IBZ for square lattice: 0 <= ky <= kx <= pi (or similar triangular sector)
    # Actually, the IBZ is the triangle (0,0), (pi,0), (pi,pi) is not right.
    # The IBZ is the triangle with vertices (0,0), (pi,0), (pi,pi) is half the square?
    # No, the IBZ is the triangle (0,0), (pi,0), (pi/2, pi/2) ?
    # Standard IBZ for square lattice: 0 <= ky <= kx <= pi is not correct.
    # The IBZ is the region 0 <= ky <= kx <= pi is not a fundamental domain.
    # The fundamental domain is the triangle (0,0), (pi,0), (pi,pi) is not correct.
    # The IBZ is the triangle (0,0), (pi,0), (pi, pi) is not right.
    # The IBZ is the triangle (0,0), (pi,0), (pi/2, pi/2) ?
    # Let's use a dense grid over the IBZ: 0 <= ky <= kx <= pi is not right.
    # The IBZ is bounded by kx=pi, ky=0, and kx=ky.
    # So 0 <= ky <= kx <= pi.
    
    kx_vals = np.linspace(0, np.pi, 10)
    ky_vals = np.linspace(0, np.pi, 10)
    k_points = []
    for kx in kx_vals:
        for ky in ky_vals:
            if ky <= kx:
                k_points.append((kx, ky))
    # Add high symmetry points explicitly
    k_points.append((0, 0))
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    
    # Remove duplicates
    k_points = list(set(k_points))
    
    def compute_gap(cell_mask):
        """
        Compute the 4th-5th TM band gap for a given binary mask.
        """
        # Calculate epsilon_G
        # mask is 1D array of size size*size
        # sum_mask_phases = sum(mask * phases, axis=1)
        sum_mask_phases = np.sum(cell_mask * phases, axis=1)
        
        # epsilon_G = eps_low * delta_G0 + (eps_high - eps_low) / size^2 * sum_mask_phases
        # delta_G0 is 1 if G=0, else 0.
        # G=0 is at index where nx=0, ny=0.
        # G_vectors are sorted. Let's find the index of (0,0).
        idx_00 = np.where((G_vectors[:, 0] == 0) & (G_vectors[:, 1] == 0))[0][0]
        
        epsilon_G = np.zeros(N_pw, dtype=complex)
        epsilon_G[idx_00] = eps_low
        epsilon_G += (eps_high - eps_low) / (size * size) * sum_mask_phases
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            # Build the matrix for this k-point
            # M_{ij} = (k + G_i) . (k + G_j) * epsilon_{G_i - G_j}
            # Note: For TM polarization, the matrix element is:
            # M_{ij} = ( (kx+Gx_i)(kx+Gx_j) + (ky+Gy_i)(ky+Gy_j) ) * epsilon_{G_i - G_j}
            # Wait, the standard form for TM is:
            # det[ (k+G)^2 delta_{G,G'} - (k+G)·(k+G') epsilon_{G-G'} ] = 0
            # Or equivalently, solve the eigenvalue problem:
            # sum_{G'} ( (k+G)·(k+G') epsilon_{G-G'} - (k+G)^2 delta_{G,G'} ) a_{G'} = -omega^2 a_G ?
            # Let's use the standard form:
            # H_{ij} = ( (k+G_i) · (k+G_j) ) * epsilon_{G_i - G_j}
            # And we want the eigenvalues of H. The band frequencies are sqrt(eigenvalues).
            # Wait, the dispersion is omega^2 = eigenvalue.
            
            k_vec = np.array([kx, ky])
            
            # (k+G_i) dot (k+G_j)
            # k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G = k_vec[None, :] + G_vectors
            # Dot product matrix
            dot_products = np.sum(k_plus_G[:, :, None] * k_plus_G[None, :, :], axis=1)
            
            # We need epsilon_{G_i - G_j}
            # G_i - G_j corresponds to vector (nx_i - nx_j, ny_i - ny_j)
            # We need to map this difference vector to the index in epsilon_G.
            # This is tricky if the difference vector is outside the cutoff.
            # However, if both G_i and G_j are within cutoff C, their difference is within 2C.
            # Our epsilon_G is only defined for G within C.
            # The plane wave method requires epsilon_G for all differences G_i - G_j.
            # If the difference is outside the cutoff, we assume epsilon_G = 0?
            # No, the Fourier series is infinite. Truncating epsilon_G to cutoff C means we ignore
            # components with |G| > C. This is the standard truncation.
            
            # To get epsilon_{G_i - G_j}, we need to find the index in G_vectors corresponding to
            # G_i - G_j. If G_i - G_j is not in G_vectors (i.e., outside cutoff), the value is 0.
            
            # Efficient way:
            # Create a dictionary of G_vector -> index
            g_dict = {}
            for idx, g in enumerate(G_vectors):
                g_dict[(g[0], g[1])] = idx
            
            # Build the matrix
            H = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i in range(N_pw):
                Gi = G_vectors[i]
                for j in range(N_pw):
                    Gj = G_vectors[j]
                    diff_G = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                    if diff_G in g_dict:
                        idx_diff = g_dict[diff_G]
                        H[i, j] = dot_products[i, j] * epsilon_G[idx_diff]
                    else:
                        H[i, j] = 0.0
            
            # Solve for eigenvalues
            # H is Hermitian (if epsilon_G is real and symmetric, which it is for real mask)
            eigenvalues = np.linalg.eigvalsh(H)
            # eigenvalues are omega^2
            # We want the 4th and 5th bands.
            # The eigenvalues are sorted in ascending order.
            # Band 1 is index 0, Band 4 is index 3, Band 5 is index 4.
            
            if len(eigenvalues) >= 5:
                omega_4 = np.sqrt(eigenvalues[3])
                omega_5 = np.sqrt(eigenvalues[4])
                
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
        
        if max_omega_4 == -np.inf or min_omega_5 == np.inf:
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = 2 * gap / midgap
        return ratio

    for R in candidate_radii:
        if time.time() > deadline:
            break
            
        # Generate cell
        cell = np.zeros((size, size), dtype=int)
        
        # Rod at (dx, dy)
        dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
        cell[dist1 <= R] = 1
        # Rod at (-dx, dy)
        dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
        cell[dist2 <= R] = 1
        # Rod at (dx, -dy)
        dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
        cell[dist3 <= R] = 1
        # Rod at (-dx, -dy)
        dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
        cell[dist4 <= R] = 1
        
        # Flatten for computation
        cell_flat = cell.ravel()
        
        # Compute gap
        gap_ratio = compute_gap(cell_flat)
        
        if gap_ratio > best_gap:
            best_gap = gap_ratio
            best_cell = cell.copy()
            
        # Optional: refine search around best R if time permits
        # For now, the discrete search is a good start.
        
    if best_cell is None:
        # Fallback to R=0.12
        R = 0.12
        cell = np.zeros((size, size), dtype=int)
        dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
        cell[dist1 <= R] = 1
        dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
        cell[dist2 <= R] = 1
        dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
        cell[dist3 <= R] = 1
        dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
        cell[dist4 <= R] = 1
        best_cell = cell
        
    return best_cell.tolist()
