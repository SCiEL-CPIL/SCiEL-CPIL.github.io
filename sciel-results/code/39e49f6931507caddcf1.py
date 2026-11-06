import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Independent block size
    half = size // 2
    block = np.zeros((half, half), dtype=int)
    
    # Initialize with a rod (like seed) or random
    # Seed: rod in center
    for i in range(half):
        for j in range(half):
            x = (j + 0.5) / half - 0.5
            y = (i + 0.5) / half - 0.5
            # Map to full cell coords?
            # Full cell center is at 0.
            # Block is top-left? 
            # Symmetry: cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
            # If we define block[i][j] for i,j in 0..15, 
            # it maps to (i,j), (i, 31-j), (31-i, j), (31-i, 31-j).
            # The center of the cell is at (15.5, 15.5).
            # The center of the block is at (7.5, 7.5).
            # Distance from cell center (15.5, 15.5) to block pixel (i, j) is:
            # dx = |i - 15.5|, dy = |j - 15.5|?
            # No, (i,j) is top-left.
            # (0,0) is far corner. (15,15) is near center.
            # Let's just use a simple pattern init.
            if i > 3 and i < 12 and j > 3 and j < 12:
                # Roughly center of the quadrant
                pass 
            # Let's just start with the seed logic adapted
            # Seed: x = (i+0.5)/32 - 0.5
            # For block i (0..15), full i is i.
            # x = (i+0.5)/32 - 0.5
            # y = (j+0.5)/32 - 0.5
            # dist < 0.2
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2*0.2:
                block[i][j] = 1

    def generate_cell(block):
        cell = np.zeros((size, size), dtype=int)
        for i in range(half):
            for j in range(half):
                val = block[i][j]
                cell[i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][j] = val
                cell[size-1-i][size-1-j] = val
        return cell

    # PWE Setup
    n_pw = 13 # 13x13 = 169 PWs
    g_max = n_pw // 2
    G_vectors = []
    for m in range(-g_max, g_max + 1):
        for n in range(-g_max, g_max + 1):
            G_vectors.append((m, n))
    G_vectors = np.array(G_vectors)
    n_pw_total = len(G_vectors)
    
    # Precompute G dot G and indices
    G_sq = G_vectors[:, 0]**2 + G_vectors[:, 1]**2
    
    # K points
    # Gamma-X: (t/2, 0), X-M: (0.5, t/2)
    k_points = []
    for t in np.linspace(0, 1, 5):
        k_points.append((t/2, 0))
        k_points.append((0.5, t/2))
    # Remove duplicates
    k_points = list(set(k_points))
    
    def compute_eps_fourier(block):
        cell = generate_cell(block)
        # cell is 32x32
        # eps = cell * eps_high + (1-cell) * eps_low
        eps = cell.astype(float) * (eps_high - eps_low) + eps_low
        
        # Compute Fourier coefficients
        # eps_G = 1/A sum eps_ij exp(-i G . r_ij)
        # A = 1
        # r_ij = ((i+0.5)/32, (j+0.5)/32)
        # G = 2 pi (m, n)
        
        # We need eps_G for G = G1 - G2
        # Max G index is g_max.
        # So we need eps_G for m, n in [-2*g_max, 2*g_max]
        
        # Vectorized computation
        i_idx = np.arange(size)
        j_idx = np.arange(size)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        
        x = (I + 0.5) / size
        y = (J + 0.5) / size
        
        # G values needed
        m_vals = np.arange(-2*g_max, 2*g_max + 1)
        n_vals = np.arange(-2*g_max, 2*g_max + 1)
        M, N = np.meshgrid(m_vals, n_vals, indexing='ij')
        
        # Phase = -2 pi (m x + n y)
        Phase = -2 * np.pi * (M * x[None, None, :, :] + N * y[None, None, :, :])
        
        # Sum eps * exp(Phase)
        # eps is 32x32. Phase is (len(m)*len(n), 32, 32)
        # This might be large. 41x41x32x32 ~ 1.7M elements. OK.
        
        eps_fourier = np.sum(eps[None, None, :, :] * np.exp(1j * Phase), axis=(2, 3)) / (size * size)
        
        # Map to dict or array indexed by (m, n)
        # We need to extract eps_G for G in G_vectors - G_vectors
        # Actually, we can just build the matrix using the full array
        
        return eps_fourier, m_vals, n_vals

    def pwe_solve(eps_fourier, m_vals, n_vals, k):
        # Build matrix
        # M_{G,G'} = (k+G) . (k+G') delta_{G,G'} - (k+G) . (k+G') / eps_{G-G'}
        
        # k is (kx, ky)
        kx, ky = k
        
        # k+G vectors
        KG = G_vectors + np.array([kx, ky])
        
        # Dot products
        # KG_i . KG_j
        KG_dot = KG @ KG.T
        
        # Delta term
        M = KG_dot * np.eye(n_pw_total)
        
        # Inverse epsilon term
        # Need 1/eps_{G-G'}
        # G-G' indices
        # m_diff = m_i - m_j
        # n_diff = n_i - n_j
        
        # m_vals index: m = -2*g_max + idx
        # We have eps_fourier indexed by m_idx, n_idx corresponding to m_vals, n_vals
        
        # Create index arrays for G_vectors
        m_G = G_vectors[:, 0]
        n_G = G_vectors[:, 1]
        
        # Diff arrays
        m_diff = m_G[:, None] - m_G[None, :]
        n_diff = n_G[:, None] - n_G[None, :]
        
        # Map to indices in m_vals, n_vals
        # m_vals starts at -2*g_max
        m_idx = m_diff + 2*g_max
        n_idx = n_diff + 2*g_max
        
        # Extract eps values
        # eps_fourier is (len(m_vals), len(n_vals))
        # Note: meshgrid indexing='ij' means eps_fourier[i, j] corresponds to m_vals[i], n_vals[j]
        eps_diff = eps_fourier[m_idx, n_idx]
        
        # Avoid division by zero
        eps_diff = np.where(eps_diff == 0, 1e-10, eps_diff)
        
        inv_eps = 1.0 / eps_diff
        
        M -= KG_dot * inv_eps
        
        # Eigenvalues
        # M is Hermitian?
        # KG_dot is real symmetric.
        # inv_eps: eps_{G-G'} = conj(eps_{G'-G}).
        # So inv_eps_{G,G'} = conj(inv_eps_{G',G}).
        # M is Hermitian.
        
        vals = np.linalg.eigvalsh(M)
        # Filter out negative or zero (numerical noise)
        vals = vals[vals > 1e-6]
        
        # We want 4th and 5th bands.
        # Ensure we have at least 5
        if len(vals) < 5:
            return None, None
            
        omega_4 = np.sqrt(vals[3])
        omega_5 = np.sqrt(vals[4])
        
        return omega_4, omega_5

    def surrogate_score(block):
        eps_fourier, m_vals, n_vals = compute_eps_fourier(block)
        
        max_omega_4 = 0
        min_omega_5 = np.inf
        
        for k in k_points:
            w4, w5 = pwe_solve(eps_fourier, m_vals, n_vals, k)
            if w4 is None: continue
            if w4 > max_omega_4:
                max_omega_4 = w4
            if w5 < min_omega_5:
                min_omega_5 = w5
                
        if min_omega_5 == np.inf:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        
        if midgap == 0:
            return 0.0
            
        return gap / midgap

    # Optimization Loop
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    current_block = block.copy()
    current_score = surrogate_score(current_block)
    best_block = current_block.copy()
    best_score = current_score
    
    no_improve_count = 0
    
    while time.time() < deadline:
        best_improvement = 0
        best_pixel = None
        best_val = None
        
        # Evaluate all flips
        for i in range(half):
            for j in range(half):
                # Flip
                old_val = current_block[i][j]
                current_block[i][j] = 1 - old_val
                
                # Score
                # Optimization: We can cache eps_fourier updates? 
                # For now, recompute.
                score = surrogate_score(current_block)
                
                # Restore
                current_block[i][j] = old_val
                
                if score > current_score + best_improvement:
                    best_improvement = score - current_score
                    best_pixel = (i, j)
                    best_val = 1 - old_val
                    
        # Apply best flip
        if best_pixel is not None:
            i, j = best_pixel
            current_block[i][j] = best_val
            current_score += best_improvement
            if current_score > best_score:
                best_score = current_score
                best_block = current_block.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Random perturbation
            if no_improve_count > 5:
                i, j = np.random.randint(0, half, 2)
                current_block[i][j] = 1 - current_block[i][j]
                current_score = surrogate_score(current_block)
                if current_score > best_score:
                    best_score = current_score
                    best_block = current_block.copy()
                no_improve_count = 0
            elif no_improve_count > 20:
                break
                
    # Final check
    final_cell = generate_cell(best_block)
    return final_cell.tolist()
