import numpy as np

def construct():
    n, m, p = 3, 3, 5
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:5]
    # This is 2x2x5. Split B into 2x2, 2x2, 2x1.
    # B cols 0,1 -> 2x2x2 (Strassen)
    # B cols 2,3 -> 2x2x2 (Strassen)
    # B col 4    -> 2x2x1 (Schoolbook)
    
    # Sub-block 1.1: A[0:2, 0:2] * B[0:2, 0:2]
    # Indices:
    # A: i in 0..1, j in 0..1 -> u_idx = i*3 + j
    # B: j in 0..1, k in 0..1 -> v_idx = j*5 + k
    # C: k in 0..1, i in 0..1 -> w_idx = k*3 + i
    
    # Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # We need to expand (A00+A11) into U terms and (B00+B11) into V terms.
    # This is not a single rank-1 term, it's a sum.
    # But in the tensor decomposition, each multiplication M_c is a rank-1 term?
    # No, Strassen reduces the number of multiplications, but the multiplications are bilinear forms.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This is NOT a rank-1 term in the original tensor basis.
    # The tensor decomposition requires $U_{\alpha, c} V_{\beta, c} W_{\gamma, c}$.
    # This implies each column $c$ is a rank-1 tensor.
    # Strassen's algorithm computes intermediate values which are linear combinations of inputs.
    # The "multiplications" in Strassen are between linear combinations.
    # So, $M_1 = U_1 \cdot V_1$.
    # $U_1$ is a vector of coefficients for A.
    # $V_1$ is a vector of coefficients for B.
    # $W_1$ is a vector of coefficients for C.
    # So yes, each Strassen multiplication corresponds to a rank-1 term in the decomposition of the *algorithm*, 
    # but does it decompose the *tensor*?
    # Yes, the tensor $T$ is the sum of rank-1 tensors corresponding to the multiplications in the bilinear algorithm.
    # If the algorithm is valid, the sum of these rank-1 tensors equals $T$.
    
    # So for M1:
    # U part: A00 + A11.
    # U[0*3+0][col] = 1
    # U[1*3+1][col] = 1
    # V part: B00 + B11.
    # V[0*5+0][col] = 1
    # V[1*5+1][col] = 1
    # W part: C11 + C12 + C21 + C22? No.
    # M1 contributes to C11, C12, C21, C22.
    # C11 += M1
    # C12 += M1? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C11 and +1 to C22.
    # W[0*3+0][col] = 1  # C11 (k=0, i=0)
    # W[1*3+1][col] = 1  # C22 (k=1, i=1)
    
    # Let's implement this logic for all 7 Strassen terms.
    
    def add_strassen_2x2x2(a_off, b_off, c_off):
        # a_off: (i_start, j_start)
        # b_off: (j_start, k_start)
        # c_off: (k_start, i_start)
        nonlocal col
        
        i0, j0 = a_off
        j1, k1 = b_off
        k2, i2 = c_off
        
        # M1
        U[i0*3+j0][col] = 1
        U[(i0+1)*3+(j0+1)][col] = 1
        V[j1*5+k1][col] = 1
        V[(j1+1)*5+(k1+1)][col] = 1
        W[k2*3+i2][col] = 1
        W[(k2+1)*3+(i2+1)][col] = 1
        col += 1
        
        # M2: (A21+A22)B11
        # A indices: (i0+1, j0), (i0+1, j0+1)
        # B indices: (j1, k1)
        # C indices: C21, C22
        U[(i0+1)*3+j0][col] = 1
        U[(i0+1)*3+(j0+1)][col] = 1
        V[j1*5+k1][col] = 1
        W[(k2)*3+(i2+1)][col] = 1 # C21
        W[(k2+1)*3+(i2+1)][col] = 1 # C22
        col += 1
        
        # M3: A11(B21-B22)
        U[i0*3+j0][col] = 1
        V[j1*5+(k1+1)][col] = 1
        V[(j1+1)*5+k1][col] = -1
        W[k2*3+i2][col] = 1 # C11? No, M3 is in C11 and C22.
        # C11 = ... + M7
        # C22 = ... + M3
        # Wait, C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M3 contributes to C12 and C22.
        W[k2*3+i2][col] = 0 # Correction
        W[k2*3+i2][col] = 1 # C12 is k=0, i=1? 
        # C indices: k is row of B, i is row of A.
        # C12 means i=0, k=1? No, C is n x p.
        # C[i, k].
        # In W, index is k*n + i.
        # C12 -> i=0, k=1 -> idx 1*3+0 = 3.
        # C22 -> i=1, k=1 -> idx 1*3+1 = 4.
        # So W[3][col] = 1, W[4][col] = 1.
        W[k2*3+i2][col] = 0
        W[(k2+1)*3+i2][col] = 1 # C12
        W[(k2+1)*3+(i2+1)][col] = 1 # C22
        col += 1
        
        # M4: A22(B12-B11)
        U[(i0+1)*3+(j0+1)][col] = 1
        V[(j1+1)*5+k1][col] = 1
        V[j1*5+k1][col] = -1
        # M4 contributes to C11 and C21.
        # C11 (i=0, k=0) -> idx 0
        # C21 (i=1, k=0) -> idx 1
        W[k2*3+i2][col] = 1
        W[k2*3+(i2+1)][col] = 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*3+j0][col] = 1
        U[i0*3+(j0+1)][col] = 1
        V[(j1+1)*5+(k1+1)][col] = 1
        # M5 contributes to C11 (-1) and C12 (+1).
        W[k2*3+i2][col] = -1
        W[(k2+1)*3+i2][col] = 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*3+j0][col] = 1
        U[i0*3+j0][col] = -1
        V[j1*5+k1][col] = 1
        V[(j1+1)*5+k1][col] = 1
        # M6 contributes to C22 (+1).
        W[(k2+1)*3+(i2+1)][col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*3+(j0+1)][col] = 1
        U[(i0+1)*3+(j0+1)][col] = -1
        V[j1*5+(k1+1)][col] = 1
        V[(j1+1)*5+(k1+1)][col] = 1
        # M7 contributes to C11 (+1).
        W[k2*3+i2][col] = 1
        col += 1

    # Call for first 2x2x2 block
    # A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    add_strassen_2x2x2((0,0), (0,0), (0,0))
    
    # Call for second 2x2x2 block
    # A[0:2, 0:2], B[0:2, 2:4], C[2:4, 0:2]
    # B cols 2,3. k starts at 2.
    add_strassen_2x2x2((0,0), (0,2), (2,0))
    
    # Block 1.3: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1)
    # Schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[(i)*3+j][col] = 1
                V[j*5+(4+k)][col] = 1
                W[(4+k)*3+i][col] = 1
                col += 1
                
    # Block 2: A[0:2, 2:3] * B[2:3, 0:5] (2x1x5)
    # A is 2x1 (col 2). B is 1x5 (row 2).
    # C is 2x5.
    # C[i, k] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*5+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
            
    # Block 3: A[2:3, 0:2] * B[0:2, 0:5] (1x2x5)
    # A is 1x2 (row 2, cols 0,1). B is 2x5.
    # C is 1x5.
    # C[2, k] = A[2, 0]B[0, k] + A[2, 1]B[1, k]
    for k in range(5):
        for j in range(2):
            U[2*3+j][col] = 1
            V[j*5+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
            
    # Block 4: A[2:3, 2:3] * B[2:3, 0:5] (1x1x5)
    # A[2,2] * B[2,k]
    for k in range(5):
        U[2*3+2][col] = 1
        V[2*5+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
