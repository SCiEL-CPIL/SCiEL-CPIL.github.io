"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 rank-7 decomposition
    # For 2x2 matrices A, B, C:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll build the 4x4x4x7 tensor decomposition using Kronecker product of
    # Strassen 2x2x2 with itself, plus schoolbook for the extra 3 columns.
    
    # First, let's build the 4x4x4 tensor via Kronecker product of Strassen.
    # The 4x4 matrix multiplication tensor T_4 = T_2 ⊗ T_2 (in appropriate sense)
    # Actually, T_4 has rank at most 7*7 = 49 via Kronecker product.
    
    # Strassen factors for 2x2x2:
    # u vectors (for A), v vectors (for B), w vectors (for C)
    # Each is in R^4 (flattened 2x2 matrix)
    
    def flatten_2x2(mat):
        """Flatten 2x2 matrix to vector [00, 01, 10, 11]"""
        return [mat[0][0], mat[0][1], mat[1][0], mat[1][1]]
    
    # Strassen 2x2 factors (u_t, v_t, w_t) for t=1..7
    # u_t: linear combination of A entries
    # v_t: linear combination of B entries  
    # w_t: linear combination of C entries (where the product contributes)
    
    # M1 = (A11 + A22)(B11 + B22) -> contributes to C11, C22
    # M2 = (A21 + A22)B11 -> contributes to C21, C22
    # M3 = A11(B21 - B22) -> contributes to C12, C22
    # M4 = A22(B12 - B11) -> contributes to C11, C21
    # M5 = (A11 + A12)B22 -> contributes to C11, C12
    # M6 = (A21 - A11)(B11 + B12) -> contributes to C22
    # M7 = (A12 - A22)(B21 + B22) -> contributes to C11
    
    # Let's define u_t as the coefficient vector for A (length 4: [A11, A12, A21, A22])
    # v_t as coefficient vector for B (length 4: [B11, B12, B21, B22])
    # w_t as coefficient vector for C (length 4: [C11, C12, C21, C22])
    
    strassen_u = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 0, 1, -1], # M3: B21 - B22
        [0, 1, -1, 0], # M4: B12 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    
    # w_t: where each M_t contributes in C
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_w = [
        [1, 0, 0, 1],   # M1: C11 + C22
        [0, 0, 1, -1],  # M2: C21 - C22
        [0, 1, 0, 1],   # M3: C12 + C22
        [1, 0, 1, 0],   # M4: C11 + C21
        [-1, 1, 0, 0],  # M5: -C11 + C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    
    # Now build the 4x4x4 decomposition via Kronecker product.
    # The 4x4 matrix A is a 2x2 block matrix of 2x2 blocks: A = [[A11_block, A12_block], [A21_block, A22_block]]
    # Similarly B and C.
    # The Kronecker product decomposition: for each pair (s, t) where s,t in 1..7,
    # we get a term in the 4x4x4 decomposition.
    
    # U vector for term (s,t): Kronecker product of u_s and u_t
    # V vector for term (s,t): Kronecker product of v_s and v_t  
    # W vector for term (s,t): Kronecker product of w_s and w_t
    
    # The 4x4 matrix flattened is [A11_11, A11_12, A12_11, A12_12, A21_11, A21_12, A22_11, A22_12]
    # Wait, let me think about the flattening convention.
    
    # For U: shape is (n*m, rank) = (16, rank)
    # U[i*m + j, col] corresponds to A[i,j]
    # So the 16 entries of A are flattened as [A00, A01, A02, A03, A10, A11, A12, A13, A20, ..., A33]
    
    # For a 2x2 block matrix where A = [[A00_2x2, A01_2x2], [A10_2x2, A11_2x2]]
    # The flattening is: [A00_00, A00_01, A01_00, A01_01, A10_00, A10_01, A11_00, A11_01]
    # Wait, that's not right either. Let me be careful.
    
    # A is 4x4, so A[i,j] for i=0..3, j=0..3
    # Flattened as [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], ..., A[3,3]]
    
    # Block structure: A[i,j] where i = 2*I + i2, j = 2*J + j2
    # So A[2*I+i2, 2*J+j2] is the (i2,j2) entry of block A[I,J]
    
    # For the Kronecker product of two 2x2x2 tensors:
    # The u vector for the 4x4 case is the Kronecker product of the u vectors for the 2x2 case.
    # But we need to be careful about the ordering.
    
    # Let's just build it directly. For each pair (s, t) in 1..7 x 1..7:
    # u_4x4[s,t] is a vector in R^16
    # v_4x4[s,t] is a vector in R^16
    # w_4x4[s,t] is a vector in R^16
    
    # The Kronecker product of u_s (length 4) and u_t (length 4) gives length 16.
    # u_s = [u_s[0], u_s[1], u_s[2], u_s[3]] corresponding to [A11, A12, A21, A22] of the 2x2 block
    # The Kronecker product u_s ⊗ u_t has entries:
    # (u_s ⊗ u_t)[i*4+j] = u_s[i] * u_t[j]
    
    # But we need to map this to the 4x4 matrix entries.
    # u_s[i] corresponds to the (i//2, i%2) block of the 2x2 matrix, which is a 2x2 submatrix.
    # So u_s[i] * u_t[j] corresponds to the entry at position ((i//2)*2 + (j//2), (i%2)*2 + (j%2)) in the 4x4 matrix?
    # No, that's not right either.
    
    # Let me think differently. The 4x4 matrix multiplication can be viewed as multiplying two 2x2 block matrices
    # where each block is 2x2. The Strassen algorithm works on the block level, and then each block multiplication
    # is itself a 2x2 matrix multiplication.
    
    # So the full decomposition is: for each Strassen term s at the block level, and each Strassen term t at the
    # sub-block level, we get a term.
    
    # Let me just enumerate all 49 terms and compute the U, V, W vectors.
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for s in range(7):
        for t in range(7):
            u_s = strassen_u[s]
            v_s = strassen_v[s]
            w_s = strassen_w[s]
            u_t = strassen_u[t]
            v_t = strassen_v[t]
            w_t = strassen_w[t]
            
            # Compute Kronecker product for U (16 entries)
            u_4x4 = [0] * 16
            v_4x4 = [0] * 16
            w_4x4 = [0] * 16
            
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    u_4x4[idx] = u_s[i] * u_t[j]
                    v_4x4[idx] = v_s[i] * v_t[j]
                    w_4x4[idx] = w_s[i] * w_t[j]
            
            U_4x4.append(u_4x4)
            V_4x4.append(v_4x4)
            W_4x4.append(w_4x4)
    
    # Now we have 49 terms for the 4x4x4 part.
    # For the 4x4x7 problem, we need to extend this.
    # The 4x4x7 tensor is: multiply 4x4 matrix A by 4x7 matrix B to get 4x7 matrix C.
    # We can view B as a 4x4 matrix B1 (first 4 columns) and a 4x3 matrix B2 (last 3 columns).
    # C = A * B = [A * B1, A * B2]
    # The first 4 columns of C are computed by the 4x4x4 decomposition above.
    # The last 3 columns of C require a 4x4x3 decomposition.
    
    # For the 4x4x3 part, we use schoolbook: rank = 4*4*3 = 48
    # Total rank = 49 + 48 = 97
    
    # Let's build the full U, V, W matrices.
    
    rank = 97
    U = [[0] * rank for _ in range(16)]  # n*m = 16
    V = [[0] * rank for _ in range(28)]  # m*p = 4*7 = 28
    W = [[0] * rank for _ in range(28)]  # p*n = 7*4 = 28
    
    col = 0
    # First 49 columns: 4x4x4 Strassen-Kronecker part
    # V and W need to be mapped: for the 4x4x4 part, B has 4 columns (indices 0-3), C has 4 rows (indices 0-3)
    # For V (m*p = 28): first 16 entries correspond to B[0:4, 0:4], next 12 entries are 0
    # For W (p*n = 28): first 16 entries correspond to C[0:4, 0:4] (flattened as C[k*n+i] = C[k, i] for k=0..3, i=0..3)
    
    for s in range(49):
        u_vec = U_4x4[s]
        v_vec = V_4x4[s]
        w_vec = W_4x4[s]
        
        for i in range(16):
            U[i][col] = u_vec[i]
        
        for i in range(16):
            V[i][col] = v_vec[i]
        # V[16:28] remains 0
        
        for i in range(16):
            W[i][col] = w_vec[i]
        # W[16:28] remains 0
        
        col += 1
    
    # Next 48 columns: 4x4x3 schoolbook part
    # For each (i, j, k) where i=0..3 (row of A), j=0..3 (col of A / row of B), k=4..6 (col of B, 0-indexed as 4,5,6)
    # C[i, k] += A[i, j] * B[j, k]
    # U[i*m + j] = 1, V[j*p + k] = 1, W[k*n + i] = 1
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
