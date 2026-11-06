import numpy as np
import time
import scipy.linalg as sla

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell maximizing the TM band gap 
    between the 4th and 5th bands using a 4x4 superlattice of circular disks.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    timeout = 220.0 # Safety margin before 230s kill

    # Pre-compute grid coordinates
    # x, y in [0, 1)
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Centers of the 4 disks in the 16x16 top-left quadrant
    # The 16x16 quadrant is divided into 4 8x8 blocks.
    # Block 0 (TL): x in [0, 0.25), y in [0, 0.25). Center: (0.125, 0.125)
    # Block 1 (TR of quadrant): x in [0.25, 0.5), y in [0, 0.25). Center: (0.375, 0.125)
    # Block 2 (BL of quadrant): x in [0, 0.25), y in [0.25, 0.5). Center: (0.125, 0.375)
    # Block 3 (BR of quadrant): x in [0.25, 0.5), y in [0.25, 0.5). Center: (0.375, 0.375)
    
    centers = np.array([
        [0.125, 0.125],
        [0.375, 0.125],
        [0.125, 0.375],
        [0.375, 0.375]
    ])
    
    # Distances from each pixel to the 4 centers
    # D[i, j, k] is distance from pixel (i,j) to center k
    D = np.zeros((size, size, 4))
    for k in range(4):
        dx = X - centers[k, 0]
        dy = Y - centers[k, 1]
        D[:, :, k] = np.sqrt(dx**2 + dy**2)

    def generate_mask(radii):
        """
        Generates a 32x32 mask based on 4 radii.
        radii: array of 4 floats.
        """
        # Mask for the 16x16 top-left quadrant
        # A pixel is dielectric (1) if it is inside ANY of the 4 disks
        # We only consider the top-left 16x16 quadrant for generation, then reflect
        mask_tl = np.zeros((16, 16), dtype=int)
        
        # Extract distance subgrid for top-left 16x16
        D_tl = D[:16, :16, :]
        
        for k in range(4):
            mask_tl |= (D_tl[:, :, k] <= radii[k]).astype(int)
            
        # Reflect to full 32x32
        mask = np.zeros((size, size), dtype=int)
        # Top-Left
        mask[:16, :16] = mask_tl
        # Top-Right (Flip X)
        mask[:16, 16:] = mask_tl[:, ::-1]
        # Bottom-Left (Flip Y)
        mask[16:, :16] = mask_tl[::-1, :]
        # Bottom-Right (Flip XY)
        mask[16:, 16:] = mask_tl[::-1, ::-1]
        
        return mask

    # Surrogate Solver Setup
    # We use a reduced plane wave expansion for speed.
    # Fast surrogate: 7x7 = 49 PWs
    # Refinement: 17x17 = 289 PWs
    
    def get_G_vectors(N):
        # N is the half-width, e.g., 3 for 7x7 (-3..3)
        G = []
        for i in range(-N, N + 1):
            for j in range(-N, N + 1):
                G.append((i, j))
        G = np.array(G)
        return G

    G_fast = get_G_vectors(3)
    G_refine = get_G_vectors(8) # 17x17: -8..8
    
    # K-points in irreducible BZ
    # (0,0) Gamma, (0.5,0) X, (0.5,0.5) M
    # Add some interior points for refinement
    k_points_fast = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5)
    ]
    
    k_points_refine = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.1, 0.4),
        (0.4, 0.1)
    ]

    def calculate_gap(mask, G, k_points):
        """
        Calculates the gap ratio using the surrogate solver.
        """
        # Compute 1/eps mask
        inv_eps_mask = np.zeros_like(mask, dtype=float)
        inv_eps_mask[mask == 1] = 1.0 / eps_high
        inv_eps_mask[mask == 0] = 1.0 / eps_low
        
        # Compute 2D FFT of 1/eps
        fft_inv_eps = np.fft.fft2(inv_eps_mask)
        fft_inv_eps = np.fft.fftshift(fft_inv_eps)
        
        # Normalize
        fft_inv_eps /= (size * size)
        
        # Build M matrix (1/eps Fourier coefficients)
        N_G = len(G)
        M = np.zeros((N_G, N_G), dtype=complex)
        
        # Pre-compute FFT indices for G differences
        # G vectors are integers. FFT index for freq f is 16 + f.
        # Since G is small, 16+G is within 0..31.
        
        for idx1, G1 in enumerate(G):
            for idx2, G2 in enumerate(G):
                dG = G1 - G2
                idx_x = 16 + int(dG[0])
                idx_y = 16 + int(dG[1])
                # Handle wrap-around if necessary, but G is small so no wrap needed for 32x32
                # However, FFT index must be mod 32.
                idx_x = idx_x % size
                idx_y = idx_y % size
                M[idx1, idx2] = fft_inv_eps[idx_x, idx_y]
        
        # Solve for each k-point
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for k in k_points:
            kx, ky = k
            # Build D matrix values (diagonal)
            D_vals = np.array([ (kx + gx)**2 + (ky + gy)**2 for gx, gy in G ])
            
            try:
                # Solve D a = omega^2 M a
                # M is Hermitian positive definite.
                eigvals, _ = sla.eigh(D_vals, M)
                eigvals = np.sort(eigvals)
                
                if len(eigvals) >= 5:
                    w4_sq = eigvals[3]
                    w5_sq = eigvals[4]
                    
                    if w4_sq > 0 and w5_sq > 0:
                        w4 = np.sqrt(w4_sq)
                        w5 = np.sqrt(w5_sq)
                        
                        max_omega_4 = max(max_omega_4, w4)
                        min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
        
        if max_omega_4 == 0.0 or min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Phase 1: Grid Search with Fast Surrogate
    # Radii range: 0.0 to 4.5 (max radius to fill 8x8 block is ~4.95, but 4.5 is safe)
    # Step: 0.25
    radii_range = np.arange(0.0, 4.51, 0.25)
    
    best_score = -1.0
    best_radii = None
    candidates = []
    
    total_evals = len(radii_range) ** 4
    # 18^4 = 104,976. Might be too slow if not optimized.
    # Let's reduce range or step.
    # Try 0.5 to 4.0, step 0.25 -> 15 steps. 15^4 = 50,625.
    # If 1ms per eval, 50s. OK.
    
    radii_range = np.arange(0.0, 4.01, 0.25)
    
    for r0 in radii_range:
        for r1 in radii_range:
            for r2 in radii_range:
                for r3 in radii_range:
                    if time.time() - start_time > timeout - 30: # Leave 30s for refinement
                        break
                    radii = np.array([r0, r1, r2, r3])
                    mask = generate_mask(radii)
                    score = calculate_gap(mask, G_fast, k_points_fast)
                    
                    if score > best_score:
                        best_score = score
                        best_radii = radii.copy()
                    
                    # Keep top candidates
                    candidates.append((score, radii))
                    
                    if time.time() - start_time > timeout - 30:
                        break
                if time.time() - start_time > timeout - 30:
                    break
            if time.time() - start_time > timeout - 30:
                break
        if time.time() - start_time > timeout - 30:
            break

    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refinement with Accurate Surrogate
    # Take top 50 candidates
    top_candidates = candidates[:50]
    
    best_score_refined = -1.0
    best_mask_refined = None
    
    for score, radii in top_candidates:
        if time.time() - start_time > timeout - 5:
            break
        
        mask = generate_mask(radii)
        # Use accurate solver
        score_acc = calculate_gap(mask, G_refine, k_points_refine)
        
        if score_acc > best_score_refined:
            best_score_refined = score_acc
            best_mask_refined = mask

    # If no refinement was done or failed, use best from phase 1
    if best_mask_refined is None:
        best_mask_refined = generate_mask(best_radii) if best_radii is not None else np.zeros((size, size), dtype=int)

    return best_mask_refined.tolist()
