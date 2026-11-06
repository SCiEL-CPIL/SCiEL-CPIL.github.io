<<<<<<< SEARCH
def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses local search to optimize the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    half = size // 2  # 16
    
    # Start with the 4-rod design
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    cell = np.zeros((size, size), dtype=int)
    dx = 0.25
    dy = 0.25
    R_rod = 0.12
    
    dist1 = np.sqrt((X - dx)**2 + (Y - dy)**2)
    cell[dist1 <= R_rod] = 1
    dist2 = np.sqrt((X + dx)**2 + (Y - dy)**2)
    cell[dist2 <= R_rod] = 1
    dist3 = np.sqrt((X - dx)**2 + (Y + dy)**2)
    cell[dist3 <= R_rod] = 1
    dist4 = np.sqrt((X + dx)**2 + (Y + dy)**2)
    cell[dist4 <= R_rod] = 1
    
    # Extract the independent quarter (top-left 16x16)
    # Due to symmetry, we only need to store and optimize this quarter
    quarter = cell[:half, :half].copy()
    
    best_score = _compute_gap(cell, n_pw=11, n_bands=6, n_kpts=5)
    best_quarter = quarter.copy()
    
    # Local search: try flipping pixels in the quarter
    # Each flip in the quarter affects 8 pixels in the full cell (due to symmetry)
    
    # Get list of independent positions
    # For D4 symmetry with transpose, the independent region is the upper triangle
    # of the top-left quarter (including diagonal)
    independent_positions = []
    for r in range(half):
        for c in range(half):
            if c <= r:  # upper triangle including diagonal
                independent_positions.append((r, c))
    
    # Also need to consider that some positions may be equivalent under symmetry
    # but for now, let's just iterate over all positions in the quarter
    
    # Actually, let me reconsider. The symmetry constraints are:
    # cell[r,c] = cell[r, size-1-c] = cell[size-1-r, c] = cell[size-1-r, size-1-c]
    # cell[r,c] = cell[c, r] = cell[c, size-1-r] = cell[size-1-r, size-1-c]
    # So the independent variables are those in the top-left triangle where r >= c
    # and r < half, c < half (to avoid double-counting with the other quarters)
    
    # Let me just use all positions in the top-left quarter and enforce symmetry after
    all_positions = [(r, c) for r in range(half) for c in range(half)]
    
    # Greedy local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r, c in all_positions:
            if time.time() > deadline:
                break
            
            # Try flipping pixel (r, c) in the quarter
            new_quarter = quarter.copy()
            new_quarter[r, c] = 1 - new_quarter[r, c]
            
            # Apply symmetry to get full cell
            new_cell = _apply_symmetry(new_quarter, size)
            
            # Evaluate
            score = _compute_gap(new_cell, n_pw=11, n_bands=6, n_kpts=5)
            
            if score > best_score:
                best_score = score
                best_quarter = new_quarter.copy()
                quarter = new_quarter
                improved = True
    
    # Final evaluation with more accurate settings
    final_cell = _apply_symmetry(best_quarter, size)
    final_score = _compute_gap(final_cell, n_pw=13, n_bands=6, n_kpts=5)
    
    return final_cell.tolist()
=======
def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses block-based exhaustive search over 4x4 grid of 8x8 sub-cells.
    Each sub-cell is defined by a 1D square-wave pattern in x and y.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    half = size // 2  # 16
    
    # Define 4x4 block structure with 8x8 sub-cells
    # Each sub-cell is determined by a 1D pattern in x (4 bits) and y (4 bits)
    # The 1D pattern is a square wave: for each of 4 segments of width 8,
    # the bit determines if it's dielectric (1) or air (0)
    
    # Precompute the 1D patterns for all 16 possible 4-bit patterns
    def make_1d_pattern(bits):
        """Create 1D pattern of length 16 from 4 bits (each bit = 4 pixels)."""
        pattern = np.zeros(16, dtype=int)
        for i in range(4):
            if bits & (1 << i):
                pattern[i*4:(i+1)*4] = 1
        return pattern
    
    # Generate all 16 possible 1D patterns
    patterns_1d = [make_1d_pattern(b) for b in range(16)]
    
    # For D4 symmetry, the x-pattern and y-pattern must be the same
    # (transpose symmetry requires cell[r,c] = cell[c,r])
    # So we only need to search over 16 possible patterns
    
    best_score = -np.inf
    best_pattern_idx = 0
    
    # Exhaustively evaluate all 16 possible symmetric patterns
    for pat_idx in range(16):
        if time.time() > deadline:
            break
        
        pattern = patterns_1d[pat_idx]
        
        # Create the full cell using this pattern for both x and y
        cell = np.zeros((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                # The pattern is defined for the first 16 pixels (one quarter)
                # Due to symmetry: cell[r,c] = pattern[r % 16] * pattern[c % 16]
                # Wait, that's not quite right. Let me think more carefully.
                
                # The 1D pattern of length 16 defines the pattern for one quarter.
                # Due to horizontal mirror: cell[r, 31-c] = cell[r, c]
                # Due to vertical mirror: cell[31-r, c] = cell[r, c]
                # Due to transpose: cell[r, c] = cell[c, r]
                
                # So the pattern for x and y should be symmetric about the center.
                # Let me redefine: the 1D pattern should be symmetric.
                
                # Actually, let me just build the cell directly.
                # For a given 4-bit pattern, the 16-pixel pattern is:
                # bits[0] -> pixels 0-3
                # bits[1] -> pixels 4-7
                # bits[2] -> pixels 8-11
                # bits[3] -> pixels 12-15
                
                # For the full 32-pixel dimension, by symmetry:
                # pixel i (0-31) maps to min(i, 31-i) which is in 0-15
                # So the full 1D pattern is: [pattern[0:16], pattern[15:-1:-1]]
                
                pass
        
        # Build full 1D pattern with symmetry
        full_1d = np.zeros(32, dtype=int)
        for i in range(32):
            idx = min(i, 31 - i)
            full_1d[i] = pattern[idx]
        
        # Create 2D cell: cell[r,c] = full_1d[r] * full_1d[c]
        cell = np.outer(full_1d, full_1d).astype(int)
        
        # Verify symmetry
        # cell should be symmetric under transpose, horizontal, vertical mirrors
        assert np.array_equal(cell, cell.T)
        assert np.array_equal(cell, cell[:, ::-1])
        assert np.array_equal(cell, cell[::-1, :])
        
        # Evaluate
        score = _compute_gap(cell, n_pw=11, n_bands=6)
        
        if score > best_score:
            best_score = score
            best_pattern_idx = pat_idx
    
    # Final evaluation with more accurate settings
    final_pattern = patterns_1d[best_pattern_idx]
    full_1d = np.zeros(32, dtype=int)
    for i in range(32):
        idx = min(i, 31 - i)
        full_1d[i] = final_pattern[idx]
    
    final_cell = np.outer(full_1d, full_1d).astype(int)
    final_score = _compute_gap(final_cell, n_pw=13, n_bands=6)
    
    return final_cell.tolist()
>>>>>>> REPLACE
