"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition using Strassen for 2x2 blocks and scalar multiplications for remainder
    rank = 22
    
    # U: (n*m) x rank = 6 x 22
    # V: (m*p) x rank = 12 x 22
    # W: (p*n) x rank = 8 x 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices mapping:
    # A (2x3): i in 0..1, j in 0..2 -> U[i*m + j]
    # B (3x4): j in 0..2, k in 0..3 -> V[j*p + k]
    # C (2x4): i in 0..1, k in 0..3 -> W[k*n + i]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, col):
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
    def add_term_neg(u_indices, v_indices, w_indices, col):
        for idx in u_indices:
            U[idx][col] -= 1
        for idx in v_indices:
            V[idx][col] -= 1
        for idx in w_indices:
            W[idx][col] -= 1

    # Strassen terms for Block 1 (C cols 0,1)
    # A_L: A[0,0]=U0, A[0,1]=U1, A[1,0]=U3, A[1,1]=U4
    # B_T1: B[0,0]=V0, B[0,1]=V1, B[1,0]=V4, B[1,1]=V5
    # C_T1: C[0,0]=W0, C[0,1]=W1, C[1,0]=W2, C[1,1]=W3
    
    # M1 = A22 * (B11 + B22) -> C12, C22
    add_term([4], [0, 5], [1, 3], 0)
    # M2 = (A12 + A22) * B22 -> C12, -C11
    add_term([1, 4], [5], [1], 1)
    add_term_neg([1, 4], [5], [0], 1) # This is not how it works, need separate columns or combine
    # Actually, the decomposition is a sum of rank-1 tensors.
    # M2 contributes +1 to C12 and -1 to C11.
    # So we need two columns? No, the tensor is sum of u v w.
    # If a term has mixed signs in W, it's still one rank-1 term if U and V are fixed.
    # Wait, rank-1 term is u_i v_j w_k.
    # If w_k has different signs for different k, that's fine.
    # So for M2:
    # U: [1, 4] have 1. V: [5] has 1. W: [1] has 1, [0] has -1.
    # This is a single rank-1 component.
    
    # Let's restart the addition logic to be precise.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # M1_1: A[1,1] * (B[0,0] + B[1,1]) -> C[0,1] + C[1,1]
    U[4][col] = 1
    V[0][col] = 1
    V[5][col] = 1
    W[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M2_1: (A[0,1] + A[1,1]) * B[1,1] -> C[0,1] - C[0,0]
    U[1][col] = 1
    U[4][col] = 1
    V[5][col] = 1
    W[1][col] = 1
    W[0][col] = -1
    col += 1
    
    # M3_1: A[0,0] * (B[0,1] - B[1,1]) -> C[1,0] + C[1,1]
    U[0][col] = 1
    V[1][col] = 1
    V[5][col] = -1
    W[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # M4_1: A[1,1] * (B[1,0] - B[0,0]) -> C[1,0] + C[0,0]
    U[4][col] = 1
    V[4][col] = 1
    V[0][col] = -1
    W[2][col] = 1
    W[0][col] = 1
    col += 1
    
    # M5_1: (A[0,0] + A[0,1]) * B[0,0] -> C[0,0] - C[1,1]
    U[0][col] = 1
    U[1][col] = 1
    V[0][col] = 1
    W[0][col] = 1
    W[3][col] = -1
    col += 1
    
    # M6_1: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1]) -> C[0,0]
    U[3][col] = 1
    U[0][col] = -1
    V[0][col] = 1
    V[1][col] = 1
    W[0][col] = 1
    col += 1
    
    # M7_1: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1]) -> C[1,1]
    U[1][col] = 1
    U[4][col] = -1
    V[4][col] = 1
    V[5][col] = 1
    W[3][col] = 1
    col += 1
    
    # Block 2 (C cols 2,3)
    # B_T2: B[0,2]=V2, B[0,3]=V3, B[1,2]=V6, B[1,3]=V7
    # C_T2: C[0,2]=W4, C[0,3]=W5, C[1,2]=W6, C[1,3]=W7
    
    # M1_2: A[1,1] * (B[0,2] + B[1,3]) -> C[0,3] + C[1,3]
    U[4][col] = 1
    V[2][col] = 1
    V[7][col] = 1
    W[5][col] = 1
    W[7][col] = 1
    col += 1
    
    # M2_2: (A[0,1] + A[1,1]) * B[1,3] -> C[0,3] - C[0,2]
    U[1][col] = 1
    U[4][col] = 1
    V[7][col] = 1
    W[5][col] = 1
    W[4][col] = -1
    col += 1
    
    # M3_2: A[0,0] * (B[0,3] - B[1,3]) -> C[1,2] + C[1,3]
    U[0][col] = 1
    V[3][col] = 1
    V[7][col] = -1
    W[6][col] = 1
    W[7][col] = 1
    col += 1
    
    # M4_2: A[1,1] * (B[1,2] - B[0,2]) -> C[1,2] + C[0,2]
    U[4][col] = 1
    V[6][col] = 1
    V[2][col] = -1
    W[6][col] = 1
    W[4][col] = 1
    col += 1
    
    # M5_2: (A[0,0] + A[0,1]) * B[0,2] -> C[0,2] - C[1,3]
    U[0][col] = 1
    U[1][col] = 1
    V[2][col] = 1
    W[4][col] = 1
    W[7][col] = -1
    col += 1
    
    # M6_2: (A[1,0] - A[0,0]) * (B[0,2] + B[0,3]) -> C[0,2]
    U[3][col] = 1
    U[0][col] = -1
    V[2][col] = 1
    V[3][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7_2: (A[0,1] - A[1,1]) * (B[1,2] + B[1,3]) -> C[1,3]
    U[1][col] = 1
    U[4][col] = -1
    V[6][col] = 1
    V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # Scalar terms
    # A_R: A[0,2]=U2, A[1,2]=U5
    # B_B: B[2,0]=V8, B[2,1]=V9, B[2,2]=V10, B[2,3]=V11
    
    # S1: A[0,2] * B[2,0] -> C[0,0]
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # S2: A[0,2] * B[2,1] -> C[0,1]
    U[2][col] = 1
    V[9][col] = 1
    W[1][col] = 1
    col += 1
    
    # S3: A[0,2] * B[2,2] -> C[0,2]
    U[2][col] = 1
    V[10][col] = 1
    W[4][col] = 1
    col += 1
    
    # S4: A[0,2] * B[2,3] -> C[0,3]
    U[2][col] = 1
    V[11][col] = 1
    W[5][col] = 1
    col += 1
    
    # S5: A[1,2] * B[2,0] -> C[1,0]
    U[5][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # S6: A[1,2] * B[2,1] -> C[1,1]
    U[5][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # S7: A[1,2] * B[2,2] -> C[1,2]
    U[5][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # S8: A[1,2] * B[2,3] -> C[1,3]
    U[5][col] = 1
    V[11][col] = 1
    W[7][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
