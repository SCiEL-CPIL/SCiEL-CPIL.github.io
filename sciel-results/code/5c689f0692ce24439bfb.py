"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition: partition 12 columns into 6 pairs,
    # each pair uses rank-7 Strassen for 2x2x2, total rank = 6*7 = 42
    n, m, p = DIMENSIONS
    # 6 pairs of columns
    num_pairs = p // 2  # = 6
    rank = num_pairs * 7  # = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 products for 2x2 matrix multiplication
    # For matrices A (2x2), B (2x2): C = A*B
    # P1 = (A11)(B12 - B22)
    # P2 = (A11 + A21)(B22)
    # P3 = (A22)(B11 - B21)
    # P4 = (A22 + A11)(B22 + B11)
    # P5 = (A11 - A22)(B21 + B22)
    # P6 = (A12 + A22)(B11)
    # P7 = (A11 + A12)(B21 - B22)
    #
    # C11 = P4 + P1 - P2 + P6
    # C12 = P3 + P7
    # C21 = P5 + P2
    # C22 = P4 - P1 + P3 + P7
    #
    # U rows: i*m+j for i in 0..1, j in 0..1 -> A[i,j]
    # V rows: j*p+k for j in 0..1, k in 0..11 -> B[j,k]
    # W rows: k*n+i for k in 0..11, i in 0..1 -> C[k,i] (output row k, col i)
    
    # Strassen product definitions: (U_coeff, V_coeffs, W_coeff)
    # Each product: U is a linear combo of A entries, V is linear combo of B entries
    # We need to express each of the 7 Strassen products as rank-1 terms
    # Actually, each Strassen product IS a product of two linear forms, so it's rank-1
    # in the tensor sense: P = u^T * a * b^T * v where a, b are the matrix entries
    
    # For each pair of columns (2k, 2k+1), we have a 2x2 submatrix B_pair
    # A is always the full 2x2 left matrix
    
    # Strassen products for one pair:
    # P1: u_A = A11, v_B = B12 - B22  => U[A11] += c, V[B12] += c, V[B22] -= c
    # P2: u_A = A11 + A21, v_B = B22
    # P3: u_A = A22, v_B = B11 - B21
    # P4: u_A = A22 + A11, v_B = B22 + B11
    # P5: u_A = A11 - A22, v_B = B21 + B22
    # P6: u_A = A12 + A22, v_B = B11
    # P7: u_A = A11 + A12, v_B = B21 - B22
    
    # Output contributions:
    # C11 = P4 + P1 - P2 + P6
    # C12 = P3 + P7
    # C21 = P5 + P2
    # C22 = P4 - P1 + P3 + P7
    
    # For pair at column offset c0 (columns c0, c0+1):
    # B[j, c0] = B11 if j=0, B21 if j=1 (column c0)
    # B[j, c0+1] = B12 if j=0, B22 if j=1 (column c0+1)
    
    # Let me define the 7 products with their U, V, W coefficients
    # U index: A[i][j] -> index i*2+j (since n*m = 4, but actually U has n*m = 4 rows)
    # Wait, U shape is (n*m, rank) = (4, rank). Row index i*m+j where m=2.
    # So A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3
    # V shape is (m*p, rank) = (24, rank). Row index j*p+k.
    # W shape is (p*n, rank) = (24, rank). Row index k*n+i.
    
    # Product definitions: (U_indices_with_coeffs, V_indices_with_coeffs, W_output_indices_with_coeffs)
    # For each product, we need:
    #   u: dict mapping U_row -> coeff
    #   v: dict mapping V_row -> coeff
    #   w: dict mapping W_row -> coeff
    
    # P1: u={0:1}, v={0*B12 - 1*B22} => for pair at c0: V[0*p + c0+1] - V[1*p + c0+1]
    #     w: C11 += 1, C22 += -1 => W[c0*2+0] += 1, W[(c0+1)*2+1] += -1
    #     Wait, C11 is row 0, col 0. W[k*n+i] where k is output row, i is output col.
    #     C[0,0] -> W[0*2+0] = W[0], C[0,1] -> W[0*2+1] = W[1]
    #     C[1,0] -> W[1*2+0] = W[2], C[1,1] -> W[1*2+1] = W[3]
    #     For pair at c0: output rows c0, c0+1
    #     C[c0, 0] -> W[c0*2+0], C[c0, 1] -> W[c0*2+1]
    #     C[c0+1, 0] -> W[(c0+1)*2+0], C[c0+1, 1] -> W[(c0+1)*2+1]
    
    for pair_idx in range(num_pairs):
        c0 = pair_idx * 2  # first column of this pair
        base = pair_idx * 7  # base rank index for this pair
        
        # P1: u={A00:1}, v={B0,c0+1: 1, B1,c0+1: -1}, w={C[c0,0]: 1, C[c0+1,1]: -1}
        U[0][base + 0] = 1
        V[0 * p + c0 + 1][base + 0] = 1
        V[1 * p + c0 + 1][base + 0] = -1
        W[c0 * n + 0][base + 0] = 1
        W[(c0 + 1) * n + 1][base + 0] = -1
        
        # P2: u={A00:1, A10:1}, v={B1,c0+1: 1}, w={C[c0,0]: -1, C[c0+1,0]: 1}
        U[0][base + 1] = 1
        U[1 * m + 0][base + 1] = 1  # A[1,0] -> row 2
        V[1 * p + c0 + 1][base + 1] = 1
        W[c0 * n + 0][base + 1] = -1
        W[(c0 + 1) * n + 0][base + 1] = 1
        
        # P3: u={A11:1}, v={B0,c0: 1, B1,c0: -1}, w={C[c0,1]: 1, C[c0+1,1]: 1}
        U[0 * m + 1][base + 2] = 1  # A[0,1] -> row 1... wait A[1,1] -> row 3
        # A[i,j] -> row i*m + j = i*2 + j
        # A[1,1] -> row 3
        U[3][base + 2] = 1
        V[0 * p + c0][base + 2] = 1
        V[1 * p + c0][base + 2] = -1
        W[c0 * n + 1][base + 2] = 1
        W[(c0 + 1) * n + 1][base + 2] = 1
        
        # P4: u={A11:1, A00:1}, v={B1,c0+1: 1, B0,c0: 1}, w={C[c0,0]: 1, C[c0+1,1]: 1}
        U[3][base + 3] = 1  # A[1,1]
        U[0][base + 3] = 1  # A[0,0]
        V[1 * p + c0 + 1][base + 3] = 1
        V[0 * p + c0][base + 3] = 1
        W[c0 * n + 0][base + 3] = 1
        W[(c0 + 1) * n + 1][base + 3] = 1
        
        # P5: u={A00:1, A11:-1}, v={B1,c0: 1, B1,c0+1: 1}, w={C[c0+1,0]: 1}
        U[0][base + 4] = 1  # A[0,0]
        U[3][base + 4] = -1  # A[1,1]
        V[1 * p + c0][base + 4] = 1
        V[1 * p + c0 + 1][base + 4] = 1
        W[(c0 + 1) * n + 0][base + 4] = 1
        
        # P6: u={A01:1, A11:1}, v={B0,c0: 1}, w={C[c0,0]: 1}
        U[0 * m + 1][base + 5] = 1  # A[0,1] -> row 1
        U[3][base + 5] = 1  # A[1,1]
        V[0 * p + c0][base + 5] = 1
        W[c0 * n + 0][base + 5] = 1
        
        # P7: u={A00:1, A01:1}, v={B1,c0: 1, B1,c0+1: -1}, w={C[c0,1]: 1, C[c0+1,1]: 1}
        U[0][base + 6] = 1  # A[0,0]
        U[1][base + 6] = 1  # A[0,1]
        V[1 * p + c0][base + 6] = 1
        V[1 * p + c0 + 1][base + 6] = -1
        W[c0 * n + 1][base + 6] = 1
        W[(c0 + 1) * n + 1][base + 6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
