"""Program for matrix multiplication 2,2,6 decomposition."""
DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using 3 Strassen blocks
    # Each 2x2x2 block has rank 7
    # B is 2x6, split into three 2x2 blocks B1, B2, B3
    # C = A * B = [A*B1 | A*B2 | A*B3]
    
    # Strassen multiplications for 2x2 matrices
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need to map these to the U, V, W format.
    # U is (n*m, rank) = (4, 21)
    # V is (m*p, rank) = (12, 21)
    # W is (p*n, rank) = (12, 21)
    
    # Indices for A (2x2):
    # 0: a, 1: b, 2: c, 3: d
    # Indices for B block 1 (2x2, cols 0,1):
    # 0: e, 1: f, 2: g, 3: h  (global V indices: 0,1,2,3)
    # Indices for B block 2 (2x2, cols 2,3):
    # 0: e, 1: f, 2: g, 3: h  (global V indices: 4,5,6,7)
    # Indices for B block 3 (2x2, cols 4,5):
    # 0: e, 1: f, 2: g, 3: h  (global V indices: 8,9,10,11)
    
    # Output C (2x6) indices (p*n = 12):
    # Block 1: 0: c00, 1: c10, 2: c01, 3: c11 (k*n+i)
    # Block 2: 4: c02, 5: c12, 6: c03, 7: c13
    # Block 3: 8: c04, 9: c14, 10: c05, 11: c15
    
    rank = 21
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    def add_strassen_block(block_idx, v_offset, w_offset):
        # block_idx is 0, 1, 2
        # v_offset is 0, 4, 8
        # w_offset is 0, 4, 8
        
        base_col = block_idx * 7
        
        # M1: (a+d)(e+h)
        # U: a(1), d(1) -> U[0, col]=1, U[3, col]=1
        # V: e(1), h(1) -> V[v_offset+0, col]=1, V[v_offset+3, col]=1
        # W: c00(+1), c10(+1), c01(+1), c11(+1)
        # C00 = M1 - M4 - M5 + M7 ? No, standard Strassen:
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # Let's define the contributions to W for each M
        # M1 contributes to C00, C11
        # M2 contributes to C10, C11
        # M3 contributes to C01, C11
        # M4 contributes to C00, C10
        # M5 contributes to C00, C01
        # M6 contributes to C11
        # M7 contributes to C00
        
        # W indices:
        # 0: c00, 1: c10, 2: c01, 3: c11
        # Global W indices: w_offset + 0, w_offset + 1, w_offset + 2, w_offset + 3
        
        # M1
        col = base_col
        U[0][col] = 1; U[3][col] = 1
        V[v_offset+0][col] = 1; V[v_offset+3][col] = 1
        W[w_offset+0][col] = 1; W[w_offset+3][col] = 1
        
        # M2
        col = base_col + 1
        U[2][col] = 1; U[3][col] = 1
        V[v_offset+0][col] = 1
        W[w_offset+1][col] = 1; W[w_offset+3][col] = -1
        
        # M3
        col = base_col + 2
        U[0][col] = 1
        V[v_offset+1][col] = 1; V[v_offset+3][col] = -1
        W[w_offset+2][col] = 1; W[w_offset+3][col] = -1
        
        # M4
        col = base_col + 3
        U[3][col] = 1
        V[v_offset+2][col] = 1; V[v_offset+0][col] = -1
        W[w_offset+0][col] = -1; W[w_offset+1][col] = 1
        
        # M5
        col = base_col + 4
        U[0][col] = 1; U[1][col] = 1
        V[v_offset+3][col] = 1
        W[w_offset+0][col] = -1; W[w_offset+2][col] = 1
        
        # M6
        col = base_col + 5
        U[2][col] = 1; U[0][col] = -1
        V[v_offset+0][col] = 1; V[v_offset+1][col] = 1
        W[w_offset+3][col] = -1
        
        # M7
        col = base_col + 6
        U[1][col] = 1; U[3][col] = -1
        V[v_offset+2][col] = 1; V[v_offset+3][col] = 1
        W[w_offset+0][col] = 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(1, 4, 4)
    add_strassen_block(2, 8, 8)

    return {"U": U, "V": V, "W": W}
