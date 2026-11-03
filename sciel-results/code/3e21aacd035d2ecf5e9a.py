"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices
    # A: 4x5 -> (3x3, 3x2; 1x3, 1x2)
    # B: 5x6 -> (3x3, 3x3; 2x3, 2x3)
    # C: 4x6 -> (3x3, 3x3; 1x3, 1x3)
    
    # We will build U, V, W lists of columns.
    # Each column corresponds to a rank-1 term.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Helper to create sparse vectors
    def make_vec(size, indices_vals):
        vec = [0] * size
        for idx, val in indices_vals:
            vec[idx] = val
        return vec

    # Laderman's algorithm for 3x3 matrices
    # Returns list of (u, v, w) for the 7 terms
    def laderman_terms():
        # Standard Laderman decomposition
        # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
        # P1 = (a+d+e)(j+m+n)
        # P2 = (b+c+f)(k+o)
        # P3 = (a+b+e)(j+k+n)
        # P4 = (c+f+i)(l+o+r)
        # P5 = (a+d+g)(j+m+p)
        # P6 = (b+e+h)(k+n+q)
        # P7 = (c+f+i)(l+o+r) -- Wait, standard Laderman has specific formulas.
        # Let's use a known set of 7 products for 3x3.
        # One common set:
        # 1. (a+d+e)(j+m+n)
        # 2. (b+c+f)(k+o)
        # 3. (a+b+e)(j+k+n)
        # 4. (c+f+i)(l+o+r)
        # 5. (a+d+g)(j+m+p)
        # 6. (b+e+h)(k+n+q)
        # 7. (d+g+h)(m+p+q)
        # This might not be correct. I will use the schoolbook for 3x3 if I'm unsure, 
        # but Laderman is rank 7.
        # Actually, I can just use schoolbook for 3x3 to be safe? No, 9 > 7.
        # I will implement a known 7-rank decomposition.
        
        # Indices in 3x3: 0..8
        # A: a0 a1 a2 / a3 a4 a5 / a6 a7 a8
        # B: b0 b1 b2 / b3 b4 b5 / b6 b7 b8
        # C: c0 c1 c2 / c3 c4 c5 / c6 c7 c8
        
        # One valid Laderman decomposition:
        # P1 = (a0+a3+a4)(b0+b3+b4)
        # P2 = (a1+a2+a5)(b1+b5)
        # P3 = (a0+a1+a4)(b0+b1+b4)
        # P4 = (a2+a5+a8)(b2+b5+b8)
        # P5 = (a0+a3+a6)(b0+b3+b6)
        # P6 = (a1+a4+a7)(b1+b4+b7)
        # P7 = (a3+a6+a7)(b3+b6+b7)
        
        terms = []
        
        # P1
        u1 = make_vec(9, [(0,1),(3,1),(4,1)])
        v1 = make_vec(9, [(0,1),(3,1),(4,1)])
        w1 = make_vec(9, [(0,1),(3,-1),(4,1),(6,-1)]) # Reconstruction coeffs
        # This is getting complicated to verify.
        # I will use schoolbook for 3x3 (rank 9) to ensure correctness, 
        # then optimize if needed. But 9 is not 7.
        # The prompt asks for rank-minimal.
        # I will use the standard Laderman 7-rank algorithm.
        
        # Correct Laderman:
        # 1. (a+d+e)(j+m+n)
        # 2. (b+c+f)(k+o)
        # 3. (a+b+e)(j+k+n)
        # 4. (c+f+i)(l+o+r)
        # 5. (a+d+g)(j+m+p)
        # 6. (b+e+h)(k+n+q)
        # 7. (d+g+h)(m+p+q)
        
        # Let's map indices:
        # a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8
        # j=0, k=1, l=2, m=3, n=4, o=5, p=6, q=7, r=8
        
        # P1
        u = make_vec(9, [(0,1),(3,1),(4,1)])
        v = make_vec(9, [(0,1),(3,1),(4,1)])
        w = make_vec(9, [(0,1),(3,1),(4,1),(6,1)]) # C00, C10, C20?
        # Reconstruction:
        # C00 = P1 - P2 - P3 + P4? No.
        # This is error-prone.
        
        # I will use schoolbook for all blocks to guarantee correctness first.
        # Rank will be 120.
        # Then I can replace 3x3 blocks with Laderman if I can verify it.
        
        # For now, I will return schoolbook.
        pass
        
    # Since I cannot easily verify Laderman coefficients in my head, 
    # and the goal is to beat 1.0 (which is 120), 
    # I should try to implement Laderman correctly.
    
    # Let's use a known correct Laderman implementation.
    
    def add_laderman_block(row_offset, col_offset):
        # A block is 3x3, B block is 3x3, C block is 3x3
        # U indices: (row_offset + i)*3 + j
        # V indices: (row_offset + j)*3 + k  -- Wait, V is indexed by (j, k) of B?
        # In the main construct, V is indexed by (j, k) where j is from A's col / B's row.
        # For the block, local j is 0..2, local k is 0..2.
        # Global V index: (global_j)*p + global_k.
        # Global W index: global_k*n + global_i.
        
        # Laderman 7 terms
        # 1. (a+d+e)(j+m+n)
        # 2. (b+c+f)(k+o)
        # 3. (a+b+e)(j+k+n)
        # 4. (c+f+i)(l+o+r)
        # 5. (a+d+g)(j+m+p)
        # 6. (b+e+h)(k+n+q)
        # 7. (d+g+h)(m+p+q)
        
        # Coeffs for C:
        # C00 = P1 - P2 - P3 + P4
        # C01 = P2 - P4
        # C02 = P4
        # C10 = P3 - P1
        # C11 = P1 - P3 - P6 + P7
        # C12 = P6 - P7
        # C20 = P5 - P3
        # C21 = P3 - P5 - P7 + P1? No.
        # C21 = P7 - P5?
        # C22 = P5 - P7
        
        # I need the exact reconstruction formula.
        # I will skip Laderman and use schoolbook for 3x3 to be safe.
        # Rank 9 for 3x3.
        # Total rank:
        # 3x3 blocks: 4 blocks? No.
        # A11 B11 (3x3): 9
        # A11 B12 (3x3): 9
        # A21 B11 (1x3 by 3x3): 9
        # A21 B12 (1x3 by 3x3): 9
        # A12 B21 (3x2 by 2x3): 18
        # A12 B22 (3x2 by 2x3): 18
        # A22 B21 (1x2 by 2x3): 6
        # A22 B22 (1x2 by 2x3): 6
        # Total: 9+9+9+9+18+18+6+6 = 84.
        
        # This is better than 120.
        
        pass

    # Schoolbook for all blocks
    # Block 11: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 12: A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 21: A[3:4, 0:3] * B[0:3, 0:3] -> C[3:4, 0:3]
    for i in range(3, 4):
        for j in range(3):
            for k in range(3):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 22: A[3:4, 0:3] * B[0:3, 3:6] -> C[3:4, 3:6]
    for i in range(3, 4):
        for j in range(3):
            for k in range(3, 6):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 13: A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]
    # A cols 3,4. B rows 3,4.
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 14: A[0:3, 3:5] * B[3:5, 3:6] -> C[0:3, 3:6]
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 6):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 23: A[3:4, 3:5] * B[3:5, 0:3] -> C[3:4, 0:3]
    for i in range(3, 4):
        for j in range(3, 5):
            for k in range(3):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Block 24: A[3:4, 3:5] * B[3:5, 3:6] -> C[3:4, 3:6]
    for i in range(3, 4):
        for j in range(3, 5):
            for k in range(3, 6):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                add_term(u, v, w)

    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in enumerate(U_cols[r]):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(V_cols[r]):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(W_cols[r]):
            if val != 0:
                W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
