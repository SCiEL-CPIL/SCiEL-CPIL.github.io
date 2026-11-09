"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition: 4 blocks of size 2x3x2
    # Each block uses Rank 11 (7 Strassen + 4 Outer Product)
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    rank = rank_per_block * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Indices in local 2x2 A/B/C:
    # A: 0(00), 1(01), 2(10), 3(11)
    # B: 0(00), 1(01), 2(10), 3(11)
    # C: 0(00), 1(01), 2(10), 3(11)
    
    # M1 = (A10+A11)(B00+B01)
    # M2 = (A10+A11)(B10+B11)
    # M3 = (A00-A11)(B00+B01)
    # M4 = A00(B00-B10)
    # M5 = A11(B11-B00)
    # M6 = (A00+A01)(B10-B11)
    # M7 = (A01-A10)(B00+B11)
    
    strassen_a = [
        [0, 0, 1, 0, 1, 0, 1], # A00 (idx 0)
        [0, 0, 0, 0, 0, 1, 1], # A01 (idx 1)
        [1, 1, 0, 0, 0, 0, -1],# A10 (idx 2)
        [1, 1, -1, 0, 1, 0, 0] # A11 (idx 3)
    ]
    strassen_b = [
        [1, 0, 1, 1, -1, 0, 1], # B00 (idx 0)
        [1, 0, 1, 0, 0, 0, 0],  # B01 (idx 1)
        [0, 1, 0, -1, 0, 1, 0], # B10 (idx 2)
        [0, 1, 0, 0, 1, -1, 1]  # B11 (idx 3)
    ]
    strassen_c = [
        [1, 0, 0, -1, 1, 0, -1], # C00 (idx 0)
        [0, 0, 1, 0, -1, -1, 0], # C01 (idx 1)
        [0, 1, -1, 0, 0, 0, 1],  # C10 (idx 2)
        [1, -1, 0, 1, 0, 1, 0]   # C11 (idx 3)
    ]
    
    for b in range(num_blocks):
        offset = b * rank_per_block
        # Block covers B columns 2b and 2b+1
        
        # 1. Strassen part (indices 0-6 in block)
        for r in range(7):
            col = offset + r
            # Map local A indices (0..3) to global U rows
            # Global A row = i * m + j. Here m=3.
            # Local A indices are for a 2x2 submatrix (cols 0,1).
            # Local 0 -> i=0, j=0 -> Global 0
            # Local 1 -> i=0, j=1 -> Global 1
            # Local 2 -> i=1, j=0 -> Global 3
            # Local 3 -> i=1, j=1 -> Global 4
            a_map = [0, 1, 3, 4]
            for la, val in enumerate(strassen_a):
                if val != 0:
                    U[a_map[la]][col] += val
            
            # Map local B indices (0..3) to global V rows
            # Local 0 -> j=0, k=2b -> Global 0 * 8 + 2b
            # Local 1 -> j=0, k=2b+1 -> Global 1 * 8? No, j*p+k.
            # V row = j * p + k.
            # Local 0: j=0, k=0 (local) -> Global j=0, k=2b. Row = 0*8 + 2b.
            # Local 1: j=0, k=1 (local) -> Global j=0, k=2b+1. Row = 0*8 + 2b+1.
            # Local 2: j=1, k=0 (local) -> Global j=1, k=2b. Row = 1*8 + 2b.
            # Local 3: j=1, k=1 (local) -> Global j=1, k=2b+1. Row = 1*8 + 2b+1.
            b_map = [2*b, 2*b+1, 8+2*b, 8+2*b+1]
            for lb, val in enumerate(strassen_b):
                if val != 0:
                    V[b_map[lb]][col] += val
            
            # Map local C indices (0..3) to global W rows
            # W row = k * n + i. n=2.
            # Local 0: k=0, i=0 -> Global k=2b, i=0. Row = 2b*2 + 0.
            # Local 1: k=1, i=0 -> Global k=2b+1, i=0. Row = (2b+1)*2 + 0.
            # Local 2: k=0, i=1 -> Global k=2b, i=1. Row = 2b*2 + 1.
            # Local 3: k=1, i=1 -> Global k=2b+1, i=1. Row = (2b+1)*2 + 1.
            c_map = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for lc, val in enumerate(strassen_c):
                if val != 0:
                    W[c_map[lc]][col] += val

        # 2. Outer Product part (indices 7-10 in block)
        # A13 (i=0, j=2) -> Global U row 2
        # A23 (i=1, j=2) -> Global U row 5
        # B31 (j=2, k=2b) -> Global V row 2*8 + 2b = 16+2b
        # B32 (j=2, k=2b+1) -> Global V row 16+2b+1
        
        # P8 = A13 * B31 -> C11 (i=0, k=2b) -> W row 4b
        U[2][offset+7] = 1
        V[16+2*b][offset+7] = 1
        W[4*b][offset+7] = 1
        
        # P9 = A13 * B32 -> C12 (i=0, k=2b+1) -> W row 4b+1
        U[2][offset+8] = 1
        V[16+2*b+1][offset+8] = 1
        W[4*b+1][offset+8] = 1
        
        # P10 = A23 * B31 -> C21 (i=1, k=2b) -> W row 4b+2
        U[5][offset+9] = 1
        V[16+2*b][offset+9] = 1
        W[4*b+2][offset+9] = 1
        
        # P11 = A23 * B32 -> C22 (i=1, k=2b+1) -> W row 4b+3
        U[5][offset+10] = 1
        V[16+2*b+1][offset+10] = 1
        W[4*b+3][offset+10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
