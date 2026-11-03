"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 2x4 * 4x8 into 8 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A * B
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Define the 7 terms for one Strassen block
    # U indices: 0,1 for A (2x2), 2,3 for B (2x2), 4,5 for C (2x2)
    # But we need to map to global indices.
    
    def add_strassen_block(offset_u, offset_v, offset_w, col_start):
        # offset_u: index of A[0,0] in U
        # offset_v: index of B[0,0] in V
        # offset_w: index of C[0,0] in W
        
        # m1
        U[offset_u][col_start] += 1
        U[offset_u + 3][col_start] += 1
        V[offset_v][col_start] += 1
        V[offset_v + 3][col_start] += 1
        W[offset_w][col_start] += 1
        W[offset_w + 3][col_start] += 1
        W[offset_w + 1][col_start] += 1
        W[offset_w + 2][col_start] += 1
        
        # m2
        U[offset_u + 2][col_start + 1] += 1
        U[offset_u + 3][col_start + 1] += 1
        V[offset_v][col_start + 1] += 1
        W[offset_w + 2][col_start + 1] += 1
        W[offset_w + 3][col_start + 1] += 1
        
        # m3
        U[offset_u][col_start + 2] += 1
        V[offset_v + 1][col_start + 2] += 1
        V[offset_v + 3][col_start + 2] -= 1
        W[offset_w + 1][col_start + 2] += 1
        W[offset_w + 3][col_start + 2] += 1
        
        # m4
        U[offset_u + 3][col_start + 3] += 1
        V[offset_v + 2][col_start + 3] += 1
        V[offset_v][col_start + 3] -= 1
        W[offset_w][col_start + 3] += 1
        W[offset_w + 2][col_start + 3] += 1
        
        # m5
        U[offset_u][col_start + 4] += 1
        U[offset_u + 1][col_start + 4] += 1
        V[offset_v + 3][col_start + 4] += 1
        W[offset_w][col_start + 4] -= 1
        W[offset_w + 1][col_start + 4] += 1
        
        # m6
        U[offset_u + 2][col_start + 5] += 1
        U[offset_u][col_start + 5] -= 1
        V[offset_v][col_start + 5] += 1
        V[offset_v + 1][col_start + 5] += 1
        W[offset_w + 3][col_start + 5] += 1
        
        # m7
        U[offset_u + 1][col_start + 6] += 1
        U[offset_u + 3][col_start + 6] -= 1
        V[offset_v + 2][col_start + 6] += 1
        V[offset_v + 3][col_start + 6] += 1
        W[offset_w][col_start + 6] += 1
        W[offset_w + 1][col_start + 6] -= 1

    # Block 1: A_L * B_T1
    # A_L is A[0:2, 0:2]. U indices 0,1,2,3.
    # B_T1 is B[0:2, 0:2]. V indices 0,1,2,3.
    # C is C[0:2, 0:2]. W indices 0,1,2,3.
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A_L * B_T2
    # B_T2 is B[0:2, 2:4]. V indices 4,5,6,7.
    # C is C[0:2, 2:4]. W indices 4,5,6,7.
    add_strassen_block(0, 4, 4, 7)
    
    # Block 3: A_L * B_T3
    # B_T3 is B[0:2, 4:6]. V indices 8,9,10,11.
    # C is C[0:2, 4:6]. W indices 8,9,10,11.
    add_strassen_block(0, 8, 8, 14)
    
    # Block 4: A_L * B_T4
    # B_T4 is B[0:2, 6:8]. V indices 12,13,14,15.
    # C is C[0:2, 6:8]. W indices 12,13,14,15.
    add_strassen_block(0, 12, 12, 21)
    
    # Block 5: A_R * B_B1
    # A_R is A[0:2, 2:4]. U indices 4,5,6,7.
    # B_B1 is B[2:4, 0:2]. V indices 16,17,18,19.
    # C is C[0:2, 0:2]. W indices 0,1,2,3.
    add_strassen_block(4, 16, 0, 28)
    
    # Block 6: A_R * B_B2
    # B_B2 is B[2:4, 2:4]. V indices 20,21,22,23.
    # C is C[0:2, 2:4]. W indices 4,5,6,7.
    add_strassen_block(4, 20, 4, 35)
    
    # Block 7: A_R * B_B3
    # B_B3 is B[2:4, 4:6]. V indices 24,25,26,27.
    # C is C[0:2, 4:6]. W indices 8,9,10,11.
    add_strassen_block(4, 24, 8, 42)
    
    # Block 8: A_R * B_B4
    # B_B4 is B[2:4, 6:8]. V indices 28,29,30,31.
    # C is C[0:2, 6:8]. W indices 12,13,14,15.
    add_strassen_block(4, 28, 12, 49)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
