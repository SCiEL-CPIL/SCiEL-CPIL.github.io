def enforce_d4_symmetry(cell):
    """
    Enforce D4 symmetry by mirroring the fundamental domain (top-left triangle)
    to the rest of the cell.
    The fundamental domain is defined by 0 <= j <= i < size//2.
    """
    N = len(cell)
    half = N // 2
    cell = np.array(cell, dtype=int)
    
    # Create a copy to avoid modifying in place if input is shared
    sym_cell = np.zeros_like(cell)
    
    # Fill the fundamental domain: i in [0, half), j in [0, i+1]
    # Note: The diagonal i=j is included.
    for i in range(half):
        for j in range(i + 1):
            val = cell[i, j]
            
            # The 8 symmetric positions for (i, j):
            # (i, j)
            # (j, i)
            # (N-1-i, N-1-j)
            # (N-1-j, N-1-i)
            # (i, N-1-j)
            # (N-1-i, j)
            # (N-1-j, i)
            # (j, N-1-i)
            
            positions = [
                (i, j),
                (j, i),
                (N-1-i, N-1-j),
                (N-1-j, N-1-i),
                (i, N-1-j),
                (N-1-i, j),
                (N-1-j, i),
                (j, N-1-i)
            ]
            
            for r, c in positions:
                sym_cell[r, c] = val
                
    return sym_cell.tolist()
