def construct():
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # 4 Strassen blocks for p=0..1, 2..3, 4..5, 6..7
    for block_idx in range(4):
        k_start = block_idx * 2
        # Add Strassen decomposition for this block
        # ... fill U, V, W using k_start ...
        col += 7
        
    # 1 Schoolbook block for p=8
    k_start = 8
    # Add Schoolbook decomposition
    # ... fill U, V, W using k_start ...
    col += 4
    
    return {"U": U, "V": V, "W": W}
