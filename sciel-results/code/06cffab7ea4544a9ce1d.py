import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Keep 10s margin for 120s limit

    # Precompute k-points for the irreducible Brillouin zone
    # Square lattice BZ: [0, pi] x [0, pi]
    # Irreducible triangle: (0,0) -> (pi,0) -> (pi,pi) -> (0,0)
    # We use 30 points as specified in the contract.
    k_points = []
    # Gamma-X: (0,0) to (pi,0)
    for t in np.linspace(0, 1, 11):
        k_points.append((math.pi * t, 0.0))
    # X-M: (pi,0) to (pi,pi)
    for t in np.linspace(0, 1, 10)[1:]: # Skip 0 to avoid duplicate X
        k_points.append((math.pi, math.pi * t))
    # M-Gamma: (pi,pi) to (0,0)
    for t in np.linspace(0, 1, 9)[1:]: # Skip 0 and 1 (M and Gamma already included)
        k_points.append((math.pi * (1 - t), math.pi * (1 - t)))
        
    # Ensure we have exactly 30 points or close to it, as per contract "30 wave vectors"
    # The contract says "30 wave vectors covering the boundary and interior".
    # My generation: 11 + 9 + 8 = 28. Let's add 2 more interior points.
    k_points.append((math.pi * 0.5, math.pi * 0.5))
    k_points.append((math.pi * 0.25, math.pi * 0.25))
    
    # Plane wave indices: 17x17 = 289
    n_pw = 17
    half = n_pw // 2
    G_indices = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel())) * (2.0 * math.pi)
    n_pw_total = len(G_vectors)
    
    # Precompute G differences indices for fast lookup
    # We need f(G_i - G_j)
    # G_i - G_j corresponds to indices (mx_i - mx_j, my_i - my_j)
    
    def compute_fourier_coeffs(cell):
        """
        Compute Fourier coefficients of 1/epsilon for the 32x32 cell.
        cell is a list of lists of 0/1.
        """
        # Convert to numpy array for speed
        cell_np = np.array(cell, dtype=np.float64)
        eps = np.where(cell_np == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Pixel centers
        i_idx = np.arange(size)
        x = (i_idx + 0.5) / size
        y = (i_idx + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        coeffs = np.zeros((n_pw, n_pw), dtype=np.complex128)
        
        for mx in range(-half, half + 1):
            for my in range(-half, half + 1):
                angle = 2.0 * math.pi * (mx * X + my * Y)
                # Sum over all pixels
                s = np.sum(inv_eps * np.exp(-1j * angle))
                coeffs[mx + half, my + half] = s / (size * size)
                
        return coeffs

    def compute_band_structure(coeffs):
        """
        Compute the band structure for the given Fourier coefficients.
        Returns max_omega_4 and min_omega_5.
        """
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        # coeffs is 17x17, indexed by [mx+half, my+half]
        # We need to access coeffs[mx_diff + half, my_diff + half]
        
        for kx, ky in k_points:
            # Vector k + G
            k_vec = np.array([kx, ky])
            KG = G_vectors + k_vec
            # |k+G|^2 is not directly needed for off-diagonals, but the matrix is:
            # M_ij = (k+G_i) . (k+G_j) * f_{G_i - G_j}
            
            # Precompute dot products (k+G_i) . (k+G_j)
            # This is an outer product of the KG vectors? No.
            # M_ij = (KG_i_x * KG_j_x + KG_i_y * KG_j_y) * f_ij
            # Let D be the matrix of dot products. D_ij = KG_i dot KG_j
            # D = KG @ KG.T ? No, that's sum over components.
            # D_ij = KG[i,0]*KG[j,0] + KG[i,1]*KG[j,1]
            # This is (KG * KG.T).T ? No.
            # D = KG.T @ KG gives sum_i KG[i,a] KG[i,b]. We want matrix of dot products.
            # D = KG @ KG.T is correct. D[i,j] = sum_a KG[i,a] KG[j,a] = KG_i dot KG_j.
            D = KG @ KG.T
            
            # Now we need to multiply D by the matrix F where F_ij = f(G_i - G_j)
            # F is a circulant-like matrix based on the difference of G indices.
            # G_indices are integers from -8 to 8.
            # G_i - G_j has components (mx_i - mx_j, my_i - my_j)
            
            # Construct F
            F = np.zeros((n_pw_total, n_pw_total), dtype=np.complex128)
            # We can do this more efficiently by iterating over differences
            # But 289x289 is small enough to loop in Python if optimized, or use numpy broadcasting.
            
            # Let's use numpy broadcasting for F construction
            # G_indices_flat is a 289x2 array of integer indices (mx, my)
            G_idx = G_vectors / (2.0 * math.pi) # Recover integer indices
            G_idx_int = G_idx.astype(int)
            
            # Differences
            # Diff[i, j, 0] = G_idx_int[i, 0] - G_idx_int[j, 0]
            # This creates a 289x289x2 array, which is 289*289*2*8 bytes ~ 1.3 MB. Acceptable.
            diffs = G_idx_int[:, np.newaxis, :] - G_idx_int[np.newaxis, :, :]
            dm_x = diffs[:, :, 0]
            dm_y = diffs[:, :, 1]
            
            # Lookup coeffs
            # coeffs is indexed by [mx+half, my+half]
            # dm_x, dm_y are in range [-8, 8]
            # So index is [dm_x + half, dm_y + half]
            F = coeffs[dm_x + half, dm_y + half]
            
            # M = D * F
            # Note: D is real, F is complex.
            M = D * F
            
            # M should be Hermitian.
            # Eigenvalues
            eigenvalues = np.linalg.eigvalsh(M)
            # omega^2 = eigenvalue
            # Filter out negative eigenvalues (numerical noise)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            omegas = np.sqrt(eigenvalues)
            
            # Sort omegas (eigvalsh returns sorted)
            # Bands 4 and 5 are indices 3 and 4 (0-indexed)
            if len(omegas) > lower_band:
                max_omega_4 = max(max_omega_4, omegas[lower_band - 1])
                min_omega_5 = min(min_omega_5, omegas[lower_band])
                
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return 2.0 * gap / midgap

    def make_cell_centered_rod(radius_px):
        """
        Create a 32x32 cell with a single centered dielectric rod.
        """
        cell = [[0] * size for _ in range(size)]
        center = (size - 1) / 2.0 # 15.5
        r_sq = radius_px * radius_px
        for i in range(size):
            for j in range(size):
                dx = i - center
                dy = j - center
                if dx*dx + dy*dy <= r_sq:
                    cell[i][j] = 1
        return cell

    def make_cell_four_rods(radius_px, offset_frac):
        """
        Create a 32x32 cell with four dielectric rods at (+/- offset, +/- offset).
        offset_frac is fraction of lattice constant (0.25 = 1/4).
        """
        cell = [[0] * size for _ in range(size)]
        center = (size - 1) / 2.0
        offset_px = offset_frac * size
        r_sq = radius_px * radius_px
        
        centers = [
            (center + offset_px, center + offset_px),
            (center - offset_px, center + offset_px),
            (center + offset_px, center - offset_px),
            (center - offset_px, center - offset_px),
        ]
        
        for i in range(size):
            for j in range(size):
                for cx, cy in centers:
                    dx = i - cx
                    dy = j - cy
                    if dx*dx + dy*dy <= r_sq:
                        cell[i][j] = 1
                        break
        return cell

    best_score = -float('inf')
    best_cell = make_cell_centered_rod(10) # Fallback

    # Search Strategy
    # 1. Centered rod radius search
    # Fill fraction for r=10px (d=20px) in 32x32: Area ~ pi*100 / 1024 ~ 0.3
    # Try radii from 8 to 14 pixels
    
    radii = [8, 9, 10, 11, 12, 13, 14]
    
    for r in radii:
        if time.time() > deadline - 20:
            break
        cell = make_cell_centered_rod(r)
        score = compute_band_structure(compute_fourier_coeffs(cell))
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # 2. Four rods search
    # Offset 0.25 (1/4) is standard.
    # Radii 4 to 8 pixels (smaller rods, 4 of them)
    # Total fill fraction similar to single rod r=10?
    # 4 * pi * 6^2 / 1024 ~ 0.44
    
    offsets = [0.25]
    radii_4 = [4, 5, 6, 7, 8]
    
    for off in offsets:
        for r in radii_4:
            if time.time() > deadline - 20:
                break
            cell = make_cell_four_rods(r, off)
            score = compute_band_structure(compute_fourier_coeffs(cell))
            if score > best_score:
                best_score = score
                best_cell = cell
                
    # 3. Local refinement around best found
    # If best was centered rod, try +/- 1 pixel radius
    # If best was 4 rods, try +/- 1 pixel radius and offset +/- 0.01
    
    # Determine which structure was best
    # We don't explicitly track, but we can re-evaluate the best cell's "type" 
    # or just do a fine grid around the parameters of the best candidate.
    # Since we don't store params with best_cell, let's just do a generic pixel-flip hill climb
    # if we have time.
    
    # Simple Hill Climb on pixels (very slow, but effective if close to optimum)
    # Budget check
    if time.time() < deadline - 5:
        current_cell = best_cell
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try flipping each pixel
            # 32x32 = 1024 pixels. Too many to flip one by one and evaluate (1024 evals * 1s = 1024s).
            # We need to be smarter.
            # Instead, let's just try a few random flips.
            for _ in range(5): # 5 random perturbations
                if time.time() > deadline - 5:
                    break
                perturbed = [row[:] for row in current_cell]
                # Flip 3 random pixels
                for _ in range(3):
                    i = np.random.randint(0, size)
                    j = np.random.randint(0, size)
                    perturbed[i][j] = 1 - perturbed[i][j]
                score = compute_band_structure(compute_fourier_coeffs(perturbed))
                if score > current_score:
                    current_score = score
                    current_cell = perturbed
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_cell = current_cell
                    break # Restart loop with new current_cell

    return best_cell

