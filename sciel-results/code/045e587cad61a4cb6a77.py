import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell using a coarse-to-fine search strategy.
    Stage 1: Optimize 4x4 blocks (16 variables).
    Stage 2: Local search on individual pixels.
    """
    start_time = time.time()
    deadline = start_time + 205.0
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    def expand_blocks_to_cell(blocks):
        """
        blocks: 4x4 numpy array of 0/1
        Returns: 32x32 numpy array
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(4):
            for j in range(4):
                if blocks[i, j] == 1:
                    cell[i*8:(i+1)*8, j*8:(j+1)*8] = 1
        # Enforce C4v symmetry
        # The block grid 4x4 is small. 
        # Symmetry of the full cell implies symmetry of the block grid?
        # Yes, if the blocks are 8x8, the symmetry axes align with block boundaries.
        # cell == cell.T => blocks == blocks.T
        # cell == cell[::-1] => blocks == blocks[::-1]
        # So we enforce symmetry on the 4x4 block grid.
        blocks = (blocks + blocks.T) // 2
        blocks = (blocks + blocks[::-1, :]) // 2
        blocks = (blocks + blocks[:, ::-1]) // 2
        # Re-expand with symmetric blocks
        cell = np.zeros((size, size), dtype=int)
        for i in range(4):
            for j in range(4):
                if blocks[i, j] == 1:
                    cell[i*8:(i+1)*8, j*8:(j+1)*8] = 1
        return cell

    def solve_bands_score(cell):
        """
        Compute the gap-midgap ratio for bands 4 and 5.
        """
        n_pw = 10
        k_pts = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.0),
            (0.25, 0.25),
            (0.5, 0.25),
        ]
        
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        
        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]
        
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))
        n_pw_actual = len(g_pairs)
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in k_pts:
            M = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            for i, (mi, ni) in enumerate(g_pairs):
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    M[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)
            
            H = (M + M.conj().T) / 2.0
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    # Stage 1: Coarse Search on 4x4 blocks
    # There are 2^16 = 65536 possible block configurations.
    # We can search a subset or use a heuristic.
    # Let's try a random search with local improvement on the 16 blocks.
    
    best_blocks = None
    best_score = -1e10
    
    # Initialize with a few random configurations
    num_init = 20
    for _ in range(num_init):
        blocks = np.random.randint(0, 2, (4, 4))
        # Enforce symmetry
        blocks = (blocks + blocks.T) // 2
        blocks = (blocks + blocks[::-1, :]) // 2
        blocks = (blocks + blocks[:, ::-1]) // 2
        cell = expand_blocks_to_cell(blocks)
        score = solve_bands_score(cell)
        if score > best_score:
            best_score = score
            best_blocks = blocks.copy()
            
    # Local search on blocks
    current_blocks = best_blocks.copy()
    current_score = best_score
    
    while time.time() < deadline - 30:
        improved = False
        # Try flipping each block
        for i in range(4):
            for j in range(4):
                if time.time() > deadline - 30:
                    break
                # Flip
                old_val = current_blocks[i, j]
                new_val = 1 - old_val
                current_blocks[i, j] = new_val
                # Enforce symmetry
                current_blocks = (current_blocks + current_blocks.T) // 2
                current_blocks = (current_blocks + current_blocks[::-1, :]) // 2
                current_blocks = (current_blocks + current_blocks[:, ::-1]) // 2
                
                cell = expand_blocks_to_cell(current_blocks)
                score = solve_bands_score(cell)
                
                if score > current_score:
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_blocks = current_blocks.copy()
                    improved = True
                else:
                    # Revert
                    current_blocks[i, j] = old_val
                    current_blocks = (current_blocks + current_blocks.T) // 2
                    current_blocks = (current_blocks + current_blocks[::-1, :]) // 2
                    current_blocks = (current_blocks + current_blocks[:, ::-1]) // 2
            if time.time() > deadline - 30:
                break
        
        if not improved:
            break

    # Stage 2: Fine Search on Pixels
    # Start from the best block configuration
    cell = expand_blocks_to_cell(best_blocks)
    
    # Local search on pixels
    # Only search pixels that are on the boundary of dielectric/air to save time?
    # Or just random pixels.
    
    current_cell = cell.copy()
    current_score = best_score
    
    # Limit iterations based on time
    max_iters = 100
    iters = 0
    
    while time.time() < deadline - 5 and iters < max_iters:
        iters += 1
        # Pick random pixel
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Flip
        old_val = current_cell[i, j]
        new_val = 1 - old_val
        current_cell[i, j] = new_val
        
        # Enforce symmetry
        # cell[i,j] -> cell[31-i, 31-j], cell[i, 31-j], cell[31-i, j], cell[j, i], etc.
        # We must flip all symmetric counterparts.
        syms = [
            (i, j),
            (31-i, 31-j),
            (i, 31-j),
            (31-i, j),
            (j, i),
            (31-j, 31-i),
            (j, 31-i),
            (31-j, i)
        ]
        
        # Set all to new_val
        for si, sj in syms:
            current_cell[si, sj] = new_val
            
        score = solve_bands_score(current_cell)
        
        if score > current_score:
            current_score = score
            if score > best_score:
                best_score = score
                best_cell = current_cell.copy()
        else:
            # Revert
            for si, sj in syms:
                current_cell[si, sj] = old_val

    # Final Result
    if 'best_cell' in locals():
        final_cell = best_cell
    else:
        final_cell = expand_blocks_to_cell(best_blocks)
        
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
        
    return result
