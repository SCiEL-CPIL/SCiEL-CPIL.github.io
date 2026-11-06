import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM 4-5 band gap.
    Uses a local search starting from dielectric-background-with-air-holes candidates.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    size = problem["pixels"] # 32
    N = size
    half = N // 2 # 16
    
    # 1. Define G-vectors for fast evaluation
    # Use 7x7 grid of G-vectors (indices -3 to 3)
    g_range = np.arange(-3, 4)
    G_vectors = np.array([(gx, gy) for gx in g_range for gy in g_range], dtype=float)
    num_G = len(G_vectors)
    
    # 2. Define k-points in irreducible BZ
    # Irreducible BZ of square lattice: 0 <= ky <= kx <= pi
    # Sample points:
    k_points = [
        (0.0, 0.0),       # Gamma
        (np.pi/2, 0.0),   # X
        (np.pi, np.pi),   # M
        (np.pi/2, np.pi/2), # Center of BZ
        (0.5*np.pi, 0.25*np.pi), # Interior
        (0.75*np.pi, 0.25*np.pi), # Interior
        (0.4*np.pi, 0.1*np.pi), # Interior
    ]
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of the dielectric constant."""
        # cell is 32x32 array of 0/1
        # epsilon = 1 + 11 * cell
        eps = 1.0 + 11.0 * cell
        coeffs = np.zeros(num_G, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            # Sum eps[x,y] * exp(-i (gx*x + gy*y) * 2pi/N)
            # x, y are 0..31
            x = np.arange(N)
            y = np.arange(N)
            phase = np.exp(-1j * 2.0 * np.pi * (gx * x[None, :] + gy * y[:, None]) / N)
            coeffs[idx] = np.sum(eps * phase)
        return coeffs

    def fast_score(cell):
        """
        Compute the gap ratio using the fast evaluator.
        Returns 0 if bands overlap or error occurs.
        """
        try:
            eps_coeffs = get_fourier_coeffs(cell)
            
            # Pre-compute 1/sqrt(eps_G) for the transformation to symmetric matrix
            # Note: eps_coeffs are complex? No, cell is real and symmetric, so coeffs should be real.
            # But due to numerical noise, take real part.
            eps_real = np.real(eps_coeffs)
            
            # Avoid division by zero (should not happen as eps >= 1)
            sqrt_eps = np.sqrt(eps_real)
            inv_sqrt_eps = 1.0 / sqrt_eps
            
            min_omega_5 = np.inf
            max_omega_4 = -np.inf
            
            for kx, ky in k_points:
                # Build matrix B
                # B_GG' = (|k+G|^2 / sqrt(eps_G) / sqrt(eps_G')) delta - eps_{G-G'} / sqrt(eps_G) / sqrt(eps_G')
                
                # Vectorized calculation of |k+G|^2
                k_vec = np.array([kx, ky])
                k_plus_G = G_vectors + k_vec
                k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
                
                # Diagonal part: |k+G|^2 / (sqrt(eps_G) sqrt(eps_G'))
                # This is outer product of (|k+G|^2 / sqrt(eps_G)) and (1/sqrt(eps_G'))?
                # No, it's diagonal.
                # Diag term: k_plus_G_sq * inv_sqrt_eps * inv_sqrt_eps?
                # Wait, B_GG = k_plus_G_sq / eps_G.
                # In transformed basis: B_GG = k_plus_G_sq / eps_G.
                # Off-diag: - eps_{G-G'} / (sqrt(eps_G) sqrt(eps_G')).
                
                # We need eps_{G-G'}.
                # G_vectors are indexed 0..num_G-1.
                # G - G' corresponds to difference of G-vectors.
                # We can pre-compute a lookup table for eps_diff?
                # Or compute on the fly.
                
                # Efficient way:
                # Create a 2D array of G differences?
                # Let's just loop G, G' for small num_G (49).
                
                B = np.zeros((num_G, num_G), dtype=float)
                
                for i in range(num_G):
                    for j in range(num_G):
                        if i == j:
                            B[i, j] = k_plus_G_sq[i] / eps_real[i]
                        else:
                            # Compute eps_{G_i - G_j}
                            diff_G = G_vectors[i] - G_vectors[j]
                            # Find index of diff_G in G_vectors?
                            # G_vectors are -3..3. Diff is -6..6.
                            # Our G_vectors only go to 3.
                            # If |diff| > 3, eps_{diff} is 0 (assuming no higher harmonics in the 7x7 basis? 
                            # No, the Fourier coefficients are defined for all G.
                            # But we only computed coeffs for G in -3..3.
                            # For G-G' outside this range, we assume 0?
                            # This is an approximation. The "fast" evaluator uses a truncated basis.
                            # This is acceptable for search.
                            
                            gx_diff = int(diff_G[0])
                            gy_diff = int(diff_G[1])
                            
                            # Check if (gx_diff, gy_diff) is in our G_vectors
                            # Find index
                            idx_diff = -1
                            for k_idx in range(num_G):
                                if G_vectors[k_idx][0] == gx_diff and G_vectors[k_idx][1] == gy_diff:
                                    idx_diff = k_idx
                                    break
                            
                            if idx_diff != -1:
                                eps_diff = eps_real[idx_diff]
                            else:
                                eps_diff = 0.0 # Approximation: ignore high order harmonics in this fast check
                
                            B[i, j] = -eps_diff * inv_sqrt_eps[i] * inv_sqrt_eps[j]
                
                # Solve eigenvalues
                eigs = np.linalg.eigvalsh(B)
                eigs = np.sort(eigs)
                
                # We want bands 4 and 5 (indices 3 and 4)
                # But we must ensure these are the correct bands.
                # In a 49 basis, we get 49 eigenvalues.
                # The first few are the lowest bands.
                # We take index 3 and 4.
                
                if len(eigs) >= 5:
                    w4 = np.sqrt(eigs[3])
                    w5 = np.sqrt(eigs[4])
                    
                    if w4 > 0 and w5 > 0:
                        if w5 > max_omega_4:
                            max_omega_4 = w5
                        if w4 < min_omega_5:
                            min_omega_5 = w4
                        
            # Calculate ratio
            if min_omega_5 == np.inf or max_omega_4 == -np.inf:
                return 0.0
            
            gap = min_omega_5 - max_omega_4
            if gap <= 0:
                return 0.0
            
            ratio = 2.0 * gap / (min_omega_5 + max_omega_4)
            return ratio
            
        except Exception:
            return 0.0

    def expand_cell(core):
        """
        Expand a 16x16 symmetric core (upper triangle) to 32x32.
        core[i][j] is defined for 0 <= i <= j < 16.
        """
        cell = np.zeros((N, N), dtype=int)
        for i in range(half):
            for j in range(i, half):
                val = core[i][j]
                # Fill symmetric positions
                cell[i, j] = val
                cell[j, i] = val
                cell[i, N-1-j] = val
                cell[N-1-j, i] = val
                cell[N-1-i, j] = val
                cell[j, N-1-i] = val
                cell[N-1-i, N-1-j] = val
                cell[N-1-j, N-1-i] = val
        return cell

    def get_core(cell):
        """Extract the 16x16 upper triangle core from a 32x32 cell."""
        core = np.zeros((half, half), dtype=int)
        for i in range(half):
            for j in range(i, half):
                core[i][j] = cell[i][j]
        return core

    # Initial Candidates
    candidates = []
    
    # 1. Central Air Hole (Dielectric Background)
    # Core radius 0.2 -> 0.2 * 32 = 6.4 pixels
    # In core coordinates (0..15), center is 7.5? No, center of 32 is 15.5.
    # In 16x16 block, center is 7.5.
    # Distance from (7.5, 7.5) < 6.4?
    # Let's create a 32x32 grid and threshold.
    x = np.arange(N) - (N - 1) / 2.0
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2)
    
    # Candidate 1: Central Hole (radius 4)
    cell1 = (r > 4.0).astype(int)
    candidates.append(cell1)
    
    # Candidate 2: Four Holes at (4,0), (0,4), etc.
    # Distance to nearest hole < 2.5
    d1 = np.sqrt((xx-4)**2 + yy**2)
    d2 = np.sqrt((xx+4)**2 + yy**2)
    d3 = np.sqrt(xx**2 + (yy-4)**2)
    d4 = np.sqrt(xx**2 + (yy+4)**2)
    min_d = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
    cell2 = (min_d > 2.5).astype(int)
    candidates.append(cell2)
    
    # Candidate 3: Cross (Dielectric along axes)
    # Air where |x|>3 and |y|>3? No, Dielectric where |x|<3 OR |y|<3.
    cell3 = ((np.abs(xx) < 3.0) | (np.abs(yy) < 3.0)).astype(int)
    candidates.append(cell3)
    
    # Candidate 4: Central Rod (Seed)
    cell4 = (r < 4.0).astype(int)
    candidates.append(cell4)
    
    # Evaluate initial candidates
    best_cell = None
    best_score = 0.0
    
    for cell in candidates:
        # Ensure symmetry (just in case)
        cell = np.array(cell)
        cell = (cell + cell.T) // 2
        cell = (cell + cell[::-1, :]) // 2
        cell = (cell + cell[:, ::-1]) // 2
        cell = (cell + cell[::-1, ::-1]) // 2
        # Fix any non-binary
        cell = (cell > 0.5).astype(int)
        
        score = fast_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    # Local Search
    # We search in the 16x16 upper triangle space
    core = get_core(best_cell)
    
    # List of independent indices
    indices = [(i, j) for i in range(half) for j in range(i, half)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias? Or just iterate.
        for i, j in indices:
            if time.time() > deadline:
                break
            
            # Flip pixel
            old_val = core[i][j]
            new_val = 1 - old_val
            core[i][j] = new_val
            
            cell = expand_cell(core)
            score = fast_score(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                improved = True
            else:
                # Revert
                core[i][j] = old_val
    
    # Final check
    if best_cell is None:
        return [[0]*N for _ in range(N)]
        
    return [[int(x) for x in row] for row in best_cell]
