DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    U_rows = n * m  # 6
    V_rows = m * p  # 30
    W_rows = p * n  # 20
    
    # Part 1: 5 Strassen blocks
    # Part 2: 20 schoolbook terms
    total_rank = 35 + 20
    U_mat = [[0] * total_rank for _ in range(U_rows)]
    V_mat = [[0] * total_rank for _ in range(V_rows)]
    W_mat = [[0] * total_rank for _ in range(W_rows)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # We define the terms as (coeff_U, U_idx, coeff_V, V_idx, coeff_W, W_idx)
    # But since we add to the same column, we just need to add coefficients.
    
    for g in range(5):
        k1 = 2 * g
        k2 = 2 * g + 1
        
        # Indices
        # U: A00(0), A01(1), A10(3), A11(4)
        # V: B00(k1), B01(k2), B10(10+k1), B11(10+k2)
        # W: C00(2*k1), C01(2*k2), C10(2*k1+1), C11(2*k2+1)
        
        # Term 0: M1
        U_mat[0][col] += 1
        U_mat[4][col] += 1
        V_mat[k1][col] += 1
        V_mat[10 + k2][col] += 1
        W_mat[2 * k1][col] += 1
        W_mat[2 * k2 + 1][col] += 1
        col += 1
        
        # Term 1: M2
        U_mat[3][col] += 1
        U_mat[4][col] += 1
        V_mat[k1][col] += 1
        W_mat[2 * k1 + 1][col] += 1
        col += 1
        
        # Term 2: M3
        U_mat[0][col] += 1
        V_mat[k2][col] += 1
        V_mat[10 + k2][col] += -1
        W_mat[2 * k2][col] += 1
        col += 1
        
        # Term 3: M4
        U_mat[4][col] += 1
        V_mat[10 + k1][col] += 1
        V_mat[k1][col] += -1
        W_mat[2 * k1][col] += -1
        W_mat[2 * k2 + 1][col] += 1
        col += 1
        
        # Term 4: M5
        U_mat[0][col] += 1
        U_mat[1][col] += 1
        V_mat[10 + k2][col] += 1
        W_mat[2 * k1][col] += 1
        col += 1
        
        # Term 5: M6
        U_mat[3][col] += 1
        U_mat[0][col] += -1
        V_mat[k1][col] += 1
        V_mat[k2][col] += 1
        W_mat[2 * k2 + 1][col] += -1
        col += 1
        
        # Term 6: M7
        U_mat[1][col] += 1
        U_mat[4][col] += -1
        V_mat[10 + k1][col] += 1
        V_mat[10 + k2][col] += 1
        W_mat[2 * k2][col] += 1
        W_mat[2 * k1][col] += -1
        col += 1
        
    # Part 2: Schoolbook for j=2
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            U_mat[i * m + 2][col] = 1
            V_mat[2 * p + k][col] = 1
            W_mat[k * n + i][col] = 1
            col += 1
            
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
