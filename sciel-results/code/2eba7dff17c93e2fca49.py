"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # Split m=3 into 2 and 1.
    # A is 2x3, B is 3x7.
    # A_L is 2x2, A_R is 2x1.
    # B_L is 2x7, B_R is 1x7.
    # C = A_L B_L + A_R B_R.
    
    # For A_L B_L (2x2x7):
    # Split B_L into three 2x2 blocks and one 2x1 block.
    # B_L1 (2x2), B_L2 (2x2), B_L3 (2x2), B_L4 (2x1).
    # Each 2x2x2 block uses Strassen (rank 7).
    # The 2x2x1 block uses schoolbook (rank 4).
    # Total rank for A_L B_L = 3*7 + 4 = 25.
    
    # For A_R B_R (2x1x7):
    # Split B_R into three 1x2 blocks and one 1x1 block.
    # B_R1 (1x2), B_R2 (1x2), B_R3 (1x2), B_R4 (1x1).
    # Each 2x1x2 block uses schoolbook (rank 2).
    # The 2x1x1 block uses schoolbook (rank 1).
    # Total rank for A_R B_R = 3*2 + 1 = 7.
    
    # Total rank = 25 + 7 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen 2x2x2
    def add_strassen(a_rows, b_cols, c_rows, c_cols):
        # a_rows: list of 2 row indices in A
        # b_cols: list of 2 col indices in B (and row indices in B_L)
        # c_rows: list of 2 row indices in C
        # c_cols: list of 2 col indices in C
        # Note: A is 2x3, B is 3x7.
        # A_L rows: 0, 1. Cols: 0, 1.
        # B_L rows: 0, 1. Cols: b_cols.
        # C rows: 0, 1. Cols: c_cols.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * (B00)
        # M3 = (A00) * (B10 - B00)
        # M4 = (A11) * (B11 - B01)
        # M5 = (A00 + A01) * (B11)
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Indices in U (A):
        # A00: a_rows[0], 0
        # A01: a_rows[0], 1
        # A10: a_rows[1], 0
        # A11: a_rows[1], 1
        
        # Indices in V (B):
        # B00: 0, b_cols[0]
        # B01: 0, b_cols[1]
        # B10: 1, b_cols[0]
        # B11: 1, b_cols[1]
        
        # Indices in W (C):
        # C00: c_cols[0], c_rows[0]
        # C01: c_cols[1], c_rows[0]
        # C10: c_cols[0], c_rows[1]
        # C11: c_cols[1], c_rows[1]
        
        # W index: k*n + i. k is col of B (row of C), i is row of A (col of C).
        # Wait, W[k*n+i, r].
        # k is index in B (0..p-1). i is index in A (0..n-1).
        # So W index = k*n + i.
        
        # C00: k=c_cols[0], i=c_rows[0] -> W[c_cols[0]*n + c_rows[0]]
        # C01: k=c_cols[1], i=c_rows[0] -> W[c_cols[1]*n + c_rows[0]]
        # C10: k=c_cols[0], i=c_rows[1] -> W[c_cols[0]*n + c_rows[1]]
        # C11: k=c_cols[1], i=c_rows[1] -> W[c_cols[1]*n + c_rows[1]]
        
        # M1
        add_block(a_rows[0], 0, c_cols[0]*n+c_rows[0], 1, 1, 1)
        add_block(a_rows[1], 0, c_cols[0]*n+c_rows[0], 1, 1, 1)
        add_block(a_rows[0], 1, c_cols[0]*n+c_rows[0], 0, 0, 0) # Placeholder, need to handle sums
        # This manual addition is error prone. I will use a generic rank-1 term addition.
        pass

    # I will implement the 32 rank decomposition directly.
    
    # A_L B_L blocks
    # Block 1: A_L (2x2) * B_L1 (2x2) -> C11 (2x2)
    # Strassen
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # C00 = M1 - M2 + M3 + M7
    # C01 = M4 + M5
    # C10 = M1 - M5 + M6 + M7
    # C11 = M2 - M3 + M4 + M6
    
    # I will write a helper to add a Strassen block.
    
    def add_strassen_block(a_r, b_c, c_r, c_c):
        nonlocal col
        # a_r: [r0, r1]
        # b_c: [c0, c1]
        # c_r: [r0, r1]
        # c_c: [c0, c1]
        
        # U indices: a_r[0]*m + 0, a_r[0]*m + 1, a_r[1]*m + 0, a_r[1]*m + 1
        u00 = a_r[0]*m + 0
        u01 = a_r[0]*m + 1
        u10 = a_r[1]*m + 0
        u11 = a_r[1]*m + 1
        
        # V indices: 0*m + b_c[0], 0*m + b_c[1], 1*m + b_c[0], 1*m + b_c[1]
        v00 = 0*m + b_c[0]
        v01 = 0*m + b_c[1]
        v10 = 1*m + b_c[0]
        v11 = 1*m + b_c[1]
        
        # W indices: c_c[0]*n + c_r[0], etc.
        w00 = c_c[0]*n + c_r[0]
        w01 = c_c[1]*n + c_r[0]
        w10 = c_c[0]*n + c_r[1]
        w11 = c_c[1]*n + c_r[1]
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; V[v00][col] = 1; W[w00][col] = 1; col+=1
        U[u11][col] = 1; V[v00][col] = 1; W[w00][col] = 1; col+=1
        U[u00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; col+=1
        U[u11][col] = 1; V[v11][col] = 1; W[w00][col] = 1; col+=1
        # Wait, Strassen uses 7 multiplications, each is a rank-1 term in the tensor.
        # The tensor term is u \otimes v \otimes w.
        # For M1, u is (1,0,1,0) in A space, v is (1,0,1,0) in B space, w is (1,1,1,1) in C space?
        # No, w is the coefficient in the output.
        # C00 += M1. So w is e00.
        # C01 += M1? No.
        # C00 = M1 - M2 + M3 + M7
        # C01 = M4 + M5
        # C10 = M1 - M5 + M6 + M7
        # C11 = M2 - M3 + M4 + M6
        
        # So M1 contributes to C00 and C10.
        # M1 = (A00+A11)(B00+B11).
        # Term 1: A00 B00 -> C00, C10
        # Term 2: A00 B11 -> C00, C10
        # Term 3: A11 B00 -> C00, C10
        # Term 4: A11 B11 -> C00, C10
        
        # This is 4 rank-1 terms for M1.
        # Total 7*4 = 28 terms? No.
        # The rank of the 2x2x2 tensor is 7.
        # This means there are 7 rank-1 tensors that sum to the multiplication tensor.
        # Each rank-1 tensor is u \otimes v \otimes w.
        # u is in R^4, v in R^4, w in R^4.
        # The 7 tensors are:
        # 1. (1,0,1,0) \otimes (1,0,1,0) \otimes (1,0,0,1) ? No.
        
        # I will use the known 7 terms for Strassen.
        
        # T1: u=(1,0,1,0), v=(1,0,1,0), w=(1,0,0,1) -> C00, C11?
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B22-B12)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M2 + M3 + M7
        # C12 = M4 + M5
        # C21 = M1 - M5 + M6 + M7
        # C22 = M2 - M3 + M4 + M6
        
        # Each Mi is a product of a linear form in A and a linear form in B.
        # So each Mi corresponds to 1 rank-1 tensor in the decomposition of the *block*.
        # But w is a vector in C space.
        # M1 contributes to C11 and C21.
        # So the tensor for M1 is u1 \otimes v1 \otimes (e11 + e21).
        # This is a rank-1 tensor? No, e11+e21 is a vector.
        # u1 \otimes v1 \otimes w1 where w1 = e11 + e21.
        # This is a single rank-1 tensor.
        
        # So we add 7 rank-1 tensors for the 2x2x2 block.
        
        # T1 (M1): u=(1,0,1,0), v=(1,0,1,0), w=(1,0,1,0) in C (C11, C21)
        U[u00][col] = 1; V[v00][col] = 1; W[w00][col] = 1; col+=1
        U[u11][col] = 1; V[v00][col] = 1; W[w00][col] = 1; col+=1
        U[u00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; col+=1
        U[u11][col] = 1; V[v11][col] = 1; W[w00][col] = 1; col+=1
        # This is 4 terms. But Strassen says 1 term.
        # The term is (A00+A11) * (B00+B11) * (C00+C10).
        # This is NOT a rank-1 tensor. It is a sum of 4.
        # So the rank is NOT 7.
        # The rank of 2x2x2 is 7.
        # This means there exist 7 rank-1 tensors.
        # I will use the explicit 7 terms from Strassen's paper.
        
        pass

    # I will fall back to schoolbook for simplicity and correctness, rank 42.
    # Then I will try to improve if I can find the 7 terms.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
