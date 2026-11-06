import numpy as np
import time
import math

def design(problem):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit
    
    size = problem['pixels']  # 32
    eps_low = problem['eps_low']  # 1
    eps_high = problem['eps_high']  # 12
    
    # --- 1. Define the search space using D4 symmetry on a coarse grid ---
    # We use a 4x4 grid of 8x8 blocks.
    # The 4x4 grid itself has D4 symmetry.
    # Independent blocks in the 4x4 grid:
    # (0,0), (0,1), (0,2), (0,3) -> row 0
    # (1,1), (1,2), (1,3) -> row 1 (since (1,0) is mirror of (0,1))
    # (2,2), (2,3) -> row 2 (since (2,0) mirror of (0,2), (2,1) mirror of (1,2))
    # (3,3) -> row 3 (mirror of (0,3))
    # Total independent blocks: 4 + 3 + 2 + 1 = 10 blocks.
    
    # Map each independent block index to its full set of positions in the 4x4 grid
    # Block coordinates in 4x4 grid: (br, bc) where br, bc in 0..3
    # Symmetries of 4x4 grid about center (1.5, 1.5):
    # (r, c) -> (3-r, 3-c) [180 rot]
    # (r, c) -> (3-r, c)   [horiz mirror]
    # (r, c) -> (r, 3-c)   [vert mirror]
    # (r, c) -> (c, r)     [diag mirror]
    # etc.
    
    indep_blocks = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 3)
    ]
    
    # For each independent block, find all symmetric positions in the 4x4 grid
    block_maps = []
    for (r, c) in indep_blocks:
        positions = set()
        # Identity
        positions.add((r, c))
        # 180 rot
        positions.add((3-r, 3-c))
        # Horiz mirror
        positions.add((3-r, c))
        # Vert mirror
        positions.add((r, 3-c))
        # Diag mirror
        positions.add((c, r))
        # Anti-diag mirror
        positions.add((3-c, 3-r))
        # 90 CW
        positions.add((c, 3-r))
        # 90 CCW
        positions.add((3-c, r))
        
        # Only keep positions that are "canonical" or just all of them?
        # Since we assign the same value to all symmetric positions, we just need the list.
        # We must ensure we don't double count in the expansion, but since we iterate
        # over independent blocks and assign to ALL their symmetric partners, it's fine.
        block_maps.append(list(positions))
        
    n_indep_blocks = len(indep_blocks) # 10
    
    # --- 2. Fast Plane Wave Solver ---
    # We need a solver that is fast enough to run ~5000-10000 times in 100s.
    # 100s / 10000 = 10ms per eval.
    # Matrix size N=25 (5x5 PW). Eigvalsh on 25x25 is ~0.1ms.
    # Building matrix: 25x25 = 625 ops. Vectorized.
    # K-points: 10 points.
    # Total: 10 * (build + eig) ~ 10 * 0.2ms = 2ms.
    # This is feasible.
    
    n_pw = 5  # 5x5 = 25 plane waves
    half = n_pw // 2
    g_vals = np.arange(-half, half + 1)
    gx, gy = np.meshgrid(g_vals, g_vals, indexing='ij')
    g_vecs = np.stack([gx.flatten(), gy.flatten()], axis=1)
    n_pw_actual = len(g_vecs)
    
    # Pre-compute G differences indices for FFT lookup
    # We need fft_inv[ga - gb] for all pairs
    # Let's pre-compute the indices into the 32x32 FFT array for all pairs
    # Actually, we can pre-compute the matrix of coefficients for a given cell?
    # No, cell changes.
    # But we can pre-compute the (ga-gb) indices.
    
    # g_diffs: (n_pw, n_pw, 2)
    g_diffs = g_vecs[:, None, :] - g_vecs[None, :, :]
    # Convert to integer indices mod 32
    # Note: g_vals are integers.
    # idx_x = (g_diffs[..., 0] % 32).astype(int)
    # idx_y = (g_diffs[..., 1] % 32).astype(int)
    idx_x = (g_diffs[..., 0] % size).astype(np.int32)
    idx_y = (g_diffs[..., 1] % size).astype(np.int32)
    
    # K-points: Gamma, X, M, and some interior
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    # To capture the gap, we need good coverage.
    # Let's use 8 k-points:
    k_points = [
        (0.0, 0.0),   # Gamma
        (0.25, 0.0),  # Mid Gamma-X
        (0.5, 0.0),   # X
        (0.5, 0.25),  # Mid X-M
        (0.5, 0.5),   # M
        (0.25, 0.25), # Center of triangle
        (0.33, 0.16), # Some interior
        (0.4, 0.2)    # Another interior
    ]
    
    def fast_score(cell_32):
        """
        cell_32: 32x32 array of 0/1
        Returns gap-midgap ratio for bands 4 and 5.
        """
        # 1/epsilon
        inv_eps = np.where(cell_32 == 1, 1.0/eps_high, 1.0/eps_low)
        
        # FFT of 1/epsilon
        # The plane wave expansion of 1/epsilon(x) = sum_G eps_G exp(i G x)
        # eps_G = (1/A) integral 1/eps(x) exp(-i G x) dx
        # Discretely: eps_G = (1/N^2) sum_{x,y} 1/eps(x,y) exp(-i G_x x - i G_y y)
        # This is exactly (1/N^2) * FFT(inv_eps)
        fft_inv = np.fft.fft2(inv_eps) / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for k in k_points:
            k_vec = np.array(k)
            # (k+G)^2 for all G
            k_plus_g = k_vec[None, :] + g_vecs
            k_sq = np.sum(k_plus_g**2, axis=1)
            
            # Matrix H_ab = eps_{G_a - G_b} * |k + G_b|^2
            # We have idx_x, idx_y for G_a - G_b
            # coeffs = fft_inv[idx_x, idx_y] -> shape (n_pw, n_pw)
            coeffs = fft_inv[idx_x, idx_y]
            
            # H = coeffs * k_sq[None, :]  (broadcast k_sq along rows)
            # Wait, H_ab = eps_{a-b} * (k+G_b)^2
            # So we multiply each column b by k_sq[b]
            matrix = coeffs * k_sq[None, :]
            
            # Solve eigenvalues
            eigs = np.linalg.eigvalsh(matrix)
            
            # Bands 4 and 5 are indices 3 and 4 (0-indexed)
            if len(eigs) > 4:
                w4_sq = eigs[3]
                w5_sq = eigs[4]
                if w4_sq > 0 and w5_sq > 0:
                    w4 = math.sqrt(w4_sq)
                    w5 = math.sqrt(w5_sq)
                    if w4 > max_omega_4:
                        max_omega_4 = w4
                    if w5 < min_omega_5:
                        min_omega_5 = w5
        
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        if gap < 0:
            return -1.0
            
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    def expand_blocks(block_vals):
        """
        block_vals: list of 10 values (0/1)
        Returns 32x32 cell
        """
        cell = np.zeros((size, size), dtype=np.uint8)
        for i, val in enumerate(block_vals):
            positions = block_maps[i]
            for (br, bc) in positions:
                # Block (br, bc) covers rows br*8 to br*8+7, cols bc*8 to bc*8+7
                r_start = br * 8
                c_start = bc * 8
                cell[r_start:r_start+8, c_start:c_start+8] = val
        return cell

    # --- 3. Search Strategy ---
    # Start with a few good guesses, then do a random search with hill climbing.
    # Since the space is 2^10 = 1024 possibilities, we can actually brute force it!
    # 1024 evaluations * 2ms = 2 seconds.
    # This is way faster than the time limit.
    # We can even use a finer grid or more k-points if we want.
    # Let's stick to the 4x4 block structure first to ensure we find a good baseline.
    
    best_score = -10.0
    best_blocks = None
    
    # Brute force all 1024 configurations
    # To make it faster, we can vectorize the expansion and scoring?
    # Maybe not easily. But 1024 loops is fine.
    
    for mask in range(1024):
        if time.time() > deadline - 10:
            break
            
        # Convert mask to block values
        block_vals = [(mask >> i) & 1 for i in range(10)]
        
        cell = expand_blocks(block_vals)
        score = fast_score(cell)
        
        if score > best_score:
            best_score = score
            best_blocks = block_vals[:]
            
    # --- 4. Refinement ---
    # The 4x4 block structure is coarse. Can we refine it?
    # We can try flipping individual pixels in the best cell found, 
    # but maintaining D4 symmetry.
    # Or, we can switch to a finer grid, e.g., 8x8 blocks of 4x4 pixels?
    # 8x8 grid has more independent pixels.
    # Let's try a local search on the pixel level starting from the best block cell.
    
    best_cell = expand_blocks(best_blocks)
    
    # Get independent pixels for the 32x32 cell
    indep_indices = []
    for i in range(16):
        for j in range(i, 32 - i):
            indep_indices.append((i, j))
            
    n_indep_px = len(indep_indices)
    
    def expand_pixels(pix_vals):
        cell = np.zeros((size, size), dtype=np.uint8)
        for idx, val in enumerate(pix_vals):
            i, j = indep_indices[idx]
            sym_positions = [
                (i, j),
                (31-i, 31-j),
                (31-i, j),
                (i, 31-j),
                (j, i),
                (31-j, 31-i),
                (j, 31-i),
                (31-j, i)
            ]
            for (ii, jj) in sym_positions:
                cell[ii, jj] = val
        return cell

    # Initialize pixel values from best_cell
    current_pix_vals = [int(best_cell[i, j]) for (i, j) in indep_indices]
    current_score = best_score
    
    # Local search: flip random pixels
    n_iter = 200
    for _ in range(n_iter):
        if time.time() > deadline - 5:
            break
            
        idx = np.random.randint(n_indep_px)
        old_val = current_pix_vals[idx]
        current_pix_vals[idx] = 1 - old_val
        
        new_cell = expand_pixels(current_pix_vals)
        new_score = fast_score(new_cell)
        
        if new_score > current_score:
            current_score = new_score
            best_cell = new_cell
        else:
            current_pix_vals[idx] = old_val
            
    return best_cell.tolist()
