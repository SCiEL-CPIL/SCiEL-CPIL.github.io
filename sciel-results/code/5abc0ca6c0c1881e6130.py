"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _fast_tm_bands(cell, n_pw=17, k_points=None):
    """
    Approximate TM band solver using plane waves.
    cell: 32x32 list of 0/1.
    n_pw: number of plane waves in each dimension (must be odd for symmetric grid).
    Returns min(omega_5) and max(omega_4) for the sampled k-points.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    # 1 = dielectric (12), 0 = air (1)
    eps[eps == 1] = 12.0
    eps[eps == 0] = 1.0
    
    # Compute Fourier coefficients of 1/eps
    # We need 1/eps in real space, then FFT
    inv_eps = 1.0 / eps
    inv_eps_fft = np.fft.fft2(inv_eps)
    # Normalize: The FFT definition in numpy is sum * exp(-2pi i ...).
    # The plane wave expansion coefficient c_G is (1/N^2) * sum ...
    # So c_G = fft2(inv_eps) / N^2
    inv_eps_fft /= (size * size)
    
    if k_points is None:
        # Sample Gamma, M, X, and one interior point
        k_points = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.25)
        ]
        
    min_omega_5 = float('inf')
    max_omega_4 = 0.0
    
    # Plane wave indices
    # Indices from -N/2 to N/2
    half = n_pw // 2
    indices = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(indices, indices, indexing='ij')
    G_vecs = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    
    for kx, ky in k_points:
        K = np.array([kx, ky])
        K_plus_G = G_vecs + K
        # Matrix M_{ij} = (K+G_i) . (K+G_j) / eps_{G_i - G_j}
        # We need to be careful with the index mapping for eps.
        # G_i is at index i. G_j is at index j.
        # G_i - G_j corresponds to frequency component.
        
        # Build the matrix
        # Dot products
        dots = np.sum(K_plus_G[:, np.newaxis, :] * K_plus_G[np.newaxis, :, :], axis=2)
        
        # Get inverse epsilon coefficients for all differences G_i - G_j
        # We need to map the difference (dx, dy) to the FFT coefficient.
        # The FFT array inv_eps_fft has indices 0..N-1.
        # A frequency component (dx, dy) in the range [-N/2, N/2] maps to:
        # if dx < 0, index = dx + N
        # if dx >= 0, index = dx
        # Same for dy.
        
        # However, G_i and G_j are from a smaller set (n_pw x n_pw).
        # The difference G_i - G_j will be in range [-n_pw, n_pw].
        # We need to extract these specific coefficients from the full size x size FFT.
        
        # To do this efficiently, we can pre-compute a lookup or just index directly.
        # Since n_pw is small (17), we can loop or vectorize the indexing.
        
        # Vectorized indexing:
        # G_i is G_vecs[i], G_j is G_vecs[j].
        # Diff = G_i - G_j.
        # We need c_{diff}.
        
        # Let's create a 2D array of differences
        # Gx_diff = Gx[:, None] - Gx[None, :]
        # Gy_diff = Gy[:, None] - Gy[None, :]
        
        # But Gx, Gy here are the flattened G_vecs.
        # Let's keep them as 2D for easier diff calculation.
        Gx_2d, Gy_2d = Gx, Gy # They are already 2D from meshgrid
        
        Dx = Gx_2d[:, None] - Gx_2d[None, :]
        Dy = Gy_2d[:, None] - Gy_2d[None, :]
        
        # Map to FFT indices
        # The FFT array size is `size` (32).
        # The frequencies are integers.
        # Index = (val % size)
        
        idx_x = Dx % size
        idx_y = Dy % size
        
        # Extract coefficients
        # inv_eps_fft is size x size
        # We need to gather values at [idx_y, idx_x]
        # Using advanced indexing
        coeffs = inv_eps_fft[idx_y, idx_x]
        
        # Build Hamiltonian
        H = dots * coeffs
        
        # Solve eigenvalues
        # H is Hermitian
        evals = np.linalg.eigvalsh(H)
        
        # We want the 4th and 5th bands (indices 3 and 4)
        # Note: The eigenvalues are omega^2.
        # The bands are ordered by omega.
        # Are there spurious states? Usually not with this formulation.
        # We take the 4th and 5th smallest positive eigenvalues.
        
        # Sometimes the first few eigenvalues might be very close to 0 or negative due to numerical noise?
        # In this formulation, eigenvalues should be positive.
        
        omega_4_sq = evals[3]
        omega_5_sq = evals[4]
        
        omega_4 = np.sqrt(omega_4_sq)
        omega_5 = np.sqrt(omega_5_sq)
        
        if omega_5 < min_omega_5:
            min_omega_5 = omega_5
        if omega_4 > max_omega_4:
            max_omega_4 = omega_4
            
    return min_omega_5, max_omega_4

def _generate_shape(shape_type, size, param):
    """
    Generate a 32x32 cell.
    shape_type: 'circle', 'square', 'diamond', 'circle_hole', 'square_hole', 'diamond_hole'
    param: size parameter (radius or half-side)
    """
    cell = np.zeros((size, size), dtype=int)
    centers = np.arange(size)
    cx, cy = np.meshgrid(centers, centers, indexing='ij')
    # Center of cell is (15.5, 15.5)
    dx = cx - 15.5
    dy = cy - 15.5
    
    if shape_type == 'circle':
        mask = (dx**2 + dy**2) < param**2
        cell[mask] = 1
    elif shape_type == 'square':
        mask = (np.abs(dx) < param) & (np.abs(dy) < param)
        cell[mask] = 1
    elif shape_type == 'diamond':
        # |dx| + |dy| < param
        mask = (np.abs(dx) + np.abs(dy)) < param
        cell[mask] = 1
    elif shape_type == 'circle_hole':
        mask = (dx**2 + dy**2) < param**2
        cell[~mask] = 1
        # Wait, if it's a hole in dielectric, the background is dielectric (1).
        # So cell is 1 everywhere, then 0 where mask is true.
        cell[:] = 1
        cell[mask] = 0
    elif shape_type == 'square_hole':
        mask = (np.abs(dx) < param) & (np.abs(dy) < param)
        cell[:] = 1
        cell[mask] = 0
    elif shape_type == 'diamond_hole':
        mask = (np.abs(dx) + np.abs(dy)) < param
        cell[:] = 1
        cell[mask] = 0
        
    # Ensure symmetry
    # The generated shapes are symmetric about the center (15.5, 15.5).
    # The problem requires cell == cell[::-1] and cell == cell.T
    # This is C4v symmetry.
    # My generation is symmetric about the center point.
    # Let's verify the flip symmetry.
    # Flip row: i -> 31-i.
    # dx_new = (31-i) - 15.5 = 15.5 - i = -dx_old.
    # Since masks use dx^2 or |dx|, they are invariant.
    # So the shapes are symmetric.
    
    return cell.tolist()

def design(problem):
    """
    Search for the best shape among a set of candidates.
    """
    start_time = time.time()
    size = problem["pixels"]
    
    candidates = []
    
    # 1. Circular Rod
    for r in np.arange(0.15, 0.35, 0.02):
        candidates.append(('circle', r))
        
    # 2. Circular Hole
    for r in np.arange(0.15, 0.35, 0.02):
        candidates.append(('circle_hole', r))
        
    # 3. Square Rod
    for s in np.arange(0.15, 0.35, 0.02):
        candidates.append(('square', s))
        
    # 4. Square Hole
    for s in np.arange(0.15, 0.35, 0.02):
        candidates.append(('square_hole', s))
        
    # 5. Diamond Rod
    for d in np.arange(0.15, 0.35, 0.02):
        candidates.append(('diamond', d))
        
    # 6. Diamond Hole
    for d in np.arange(0.15, 0.35, 0.02):
        candidates.append(('diamond_hole', d))
        
    best_score = -float('inf')
    best_cell = None
    
    for shape_type, param in candidates:
        if time.time() - start_time > 100: # Safety margin
            break
            
        cell = _generate_shape(shape_type, size, param)
        
        # Evaluate with fast solver
        try:
            min_omega_5, max_omega_4 = _fast_tm_bands(cell, n_pw=17)
            if max_omega_4 > 0:
                score = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
            else:
                score = -1.0 # Overlap or invalid
        except Exception:
            score = -1.0
            
        if score > best_score:
            best_score = score
            best_cell = cell
            
    # If no candidate found or all failed, return seed
    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
        
    return best_cell
# EVOLVE-BLOCK-END
