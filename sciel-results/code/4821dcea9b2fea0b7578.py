"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    # A_right: rows 0,1 of A, cols 0,1 (indices in U: 0,1,3,4)
    # B_top: rows 0,1 of B (indices in V: 0..9, 10..19)
    # C: all 20 output elements
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # A indices in U: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        # B indices in V: B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=10+k0, B[1,k1]=10+k1
        # C indices in W: C[0,k0]=2*k0, C[0,k1]=2*k1, C[1,k0]=2*k0+1, C[1,k1]=2*k1+1
        
        # M1 = (A00 + A11) * (B00 + B11)
        U_mat[0][idx] += 1
        U_mat[4][idx] += 1
        V_mat[k0][idx] += 1
        V_mat[10 + k1][idx] += 1
        W_mat[2 * k0][idx] += 1
        W_mat[2 * k1 + 1][idx] += 1
        idx += 1
        
        # M2 = (A10 + A11) * B00
        U_mat[3][idx] += 1
        U_mat[4][idx] += 1
        V_mat[k0][idx] += 1
        W_mat[2 * k0 + 1][idx] += 1
        W_mat[2 * k1 + 1][idx] -= 1
        idx += 1
        
        # M3 = A00 * (B01 - B11)
        U_mat[0][idx] += 1
        V_mat[k1][idx] += 1
        V_mat[10 + k1][idx] -= 1
        W_mat[2 * k1][idx] += 1
        W_mat[2 * k1 + 1][idx] += 1
        idx += 1
        
        # M4 = A11 * (B10 - B00)
        U_mat[4][idx] += 1
        V_mat[10 + k0][idx] += 1
        V_mat[k0][idx] -= 1
        W_mat[2 * k0][idx] += 1
        W_mat[2 * k0 + 1][idx] += 1
        idx += 1
        
        # M5 = (A00 + A01) * B11
        U_mat[0][idx] += 1
        U_mat[1][idx] += 1
        V_mat[10 + k1][idx] += 1
        W_mat[2 * k0][idx] -= 1
        W_mat[2 * k1][idx] += 1
        idx += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U_mat[3][idx] += 1
        U_mat[0][idx] -= 1
        V_mat[k0][idx] += 1
        V_mat[k1][idx] += 1
        W_mat[2 * k1 + 1][idx] += 1
        idx += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U_mat[1][idx] += 1
        U_mat[4][idx] -= 1
        V_mat[10 + k0][idx] += 1
        V_mat[10 + k1][idx] += 1
        W_mat[2 * k0][idx] += 1
        idx += 1
    
    # Part 2: Schoolbook for 2x1 x 1x10 (A col 2, B row 2)
    # A[0,2] index in U: 2, A[1,2] index in U: 5
    # B[2,k] index in V: 20 + k
    # C[i,k] index in W: 2*k + i
    for i in range(2):
        for k in range(10):
            U_mat[i * 3 + 2][idx] += 1
            V_mat[20 + k][idx] += 1
            W_mat[2 * k + i][idx] += 1
            idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
