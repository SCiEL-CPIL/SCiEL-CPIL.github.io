def construct():
    n, m, p = 4, 4, 7
    rank = 49 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 2x2 Strassen local vectors
    # u_s: list of (i, j, val)
    # v_s: list of (j, k, val)
    # w_s: list of (k, i, val)
    
    sU = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    sV = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    sW = [
        [(0,0,1), (1,1,1)], # M1 -> C00, C11
        [(1,0,1), (1,1,-1)],# M2 -> C10, C11
        [(0,1,1), (1,1,1)], # M3 -> C01, C11
        [(0,0,1), (1,0,1)], # M4 -> C00, C10
        [(0,0,-1), (0,1,1)],# M5 -> C00, C01
        [(1,1,1)],          # M6 -> C11
        [(0,0,1)]           # M7 -> C00
    ]
    
    # Top-level Strassen specs
    # A_blocks: list of (block_row, block_col, sign)
    # B_blocks: list of (block_row, block_col, sign)
    # C_blocks: list of (block_row, block_col, sign)
    
    # Note: C blocks are just for mapping W. The W mapping depends on which C block the result goes to.
    # But in Strassen, the result M_l is a 2x2 matrix.
    # The global C is formed by linear combinations of M_l.
    # However, the W vector in the decomposition corresponds to the *output* of the product M_l.
    # The position of this output in the global C is determined by the Strassen reconstruction formula.
    # Wait. The tensor decomposition is $\sum u \otimes v \otimes w$.
    # $w$ is a vector of length $p \cdot n = 28$.
    # The non-zeros in $w$ indicate which entries of C receive the contribution.
    # For a specific term $(l,s)$, the contribution goes to the C entries corresponding to the 2x2 block(s) where $M_l$ is used.
    # $M_l$ contributes to multiple C blocks.
    # E.g., $M_1$ contributes to $C_{00}$ and $C_{11}$.
    # So the local W vector $w_s$ (which has non-zeros at local C positions) must be mapped to *all* global C positions where $M_l$ is used, with the appropriate signs.
    
    # So for each $l$, we need to know which C blocks it contributes to and with what sign.
    
    C_specs = [
        # M1
        [(0,0,1), (1,1,1)],
        # M2
        [(1,0,1), (1,1,-1)],
        # M3
        [(0,1,1), (1,1,1)],
        # M4
        [(0,0,1), (1,0,1)],
        # M5
        [(0,0,-1), (0,1,1)],
        # M6
        [(1,1,1)],
        # M7
        [(0,0,1)]
    ]
    
    A_specs = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    B_specs = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    for l in range(7):
        for s in range(7):
            # Process U
            for i, j, val in sU[s]:
                for br, bc, sign in A_specs[l]:
                    # Global A index: (br*2+i)*4 + (bc*2+j)
                    gu = (br * 2 + i) * 4 + (bc * 2 + j)
                    U[gu][col] += val * sign
            
            # Process V
            for j, k, val in sV[s]:
                for br, bc, sign in B_specs[l]:
                    # Global B index: (br*2+j)*7 + (bc*2+k)
                    gv = (br * 2 + j) * 7 + (bc * 2 + k)
                    V[gv][col] += val * sign
            
            # Process W
            for k, i, val in sW[s]:
                for cr, cc, sign in C_specs[l]:
                    # Global C index: (cc*2+k)*4 + (cr*2+i)
                    gw = (cc * 2 + k) * 4 + (cr * 2 + i)
                    W[gw][col] += val * sign
            
            col += 1
            
    # Part 2: Schoolbook 4x4 * 4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # Global U index: i*4+j
                gu = i * 4 + j
                # Global V index: j*7 + (4+k)  (cols 4,5,6 of B)
                gv = j * 7 + (4 + k)
                # Global W index: (4+k)*4 + i
                gw = (4 + k) * 4 + i
                
                U[gu][col] = 1
                V[gv][col] = 1
                W[gw][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
