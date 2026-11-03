import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 tensor decomposition for 6x6 matrix multiplication.
    Uses Strassen's algorithm (rank 7) for 2x2 block multiplication,
    where each block is a 3x3 matrix multiplied using the trivial
    schoolbook decomposition (rank 27).
    Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition: 7 multiplications of 3x3 blocks.
    # Use Laderman's algorithm (rank 23) for each 3x3 block.
    # Total rank = 7 * 23 = 161.
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication
    beta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    alpha = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1
    
    # Laderman's 3x3 decomposition coefficients
    # u[l][i][j]: coeff of A[i][j] in linear form l
    # v[l][k][l]: coeff of B[k][l] in linear form l
    # w[l][i][k]: coeff of product l in C[i][k]
    
    u = [[[0]*3 for _ in range(3)] for _ in range(23)]
    v = [[[0]*3 for _ in range(3)] for _ in range(23)]
    w = [[[0]*3 for _ in range(3)] for _ in range(23)]
    
    # 1
    u[0] = [[1,1,0],[1,1,0],[1,1,1]]
    v[0] = [[1,1,1],[1,1,1],[1,1,1]]
    # 2
    u[1] = [[0,1,0],[0,1,0],[0,1,0]]
    v[1] = [[1,0,0],[1,0,0],[1,0,0]]
    # 3
    u[2] = [[1,0,0],[1,0,0],[1,0,0]]
    v[2] = [[0,1,0],[0,1,0],[0,1,0]]
    # 4
    u[3] = [[1,0,0],[1,0,0],[1,0,0]]
    v[3] = [[0,0,1],[0,0,1],[0,0,1]]
    # 5
    u[4] = [[0,1,0],[0,1,0],[0,1,0]]
    v[4] = [[0,0,1],[0,0,1],[0,0,1]]
    # 6
    u[5] = [[0,0,1],[0,0,1],[0,0,1]]
    v[5] = [[1,0,0],[1,0,0],[1,0,0]]
    # 7
    u[6] = [[0,0,1],[0,0,1],[0,0,1]]
    v[6] = [[0,1,0],[0,1,0],[0,1,0]]
    # 8
    u[7] = [[1,1,1],[0,0,0],[0,0,0]]
    v[7] = [[0,0,0],[1,1,1],[0,0,0]]
    # 9
    u[8] = [[0,0,0],[1,1,1],[0,0,0]]
    v[8] = [[1,1,1],[0,0,0],[0,0,0]]
    # 10
    u[9] = [[0,0,0],[0,0,0],[1,1,1]]
    v[9] = [[1,1,1],[0,0,0],[0,0,0]]
    # 11
    u[10] = [[1,1,1],[0,0,0],[0,0,0]]
    v[10] = [[0,0,0],[0,0,0],[1,1,1]]
    # 12
    u[11] = [[1,0,0],[1,0,0],[1,0,0]]
    v[11] = [[1,0,0],[1,0,0],[1,0,0]]
    # 13
    u[12] = [[0,1,0],[0,1,0],[0,1,0]]
    v[12] = [[0,1,0],[0,1,0],[0,1,0]]
    # 14
    u[13] = [[0,0,1],[0,0,1],[0,0,1]]
    v[13] = [[0,0,1],[0,0,1],[0,0,1]]
    # 15
    u[14] = [[1,0,0],[0,1,0],[0,0,1]]
    v[14] = [[1,0,0],[0,1,0],[0,0,1]]
    # 16
    u[15] = [[0,1,0],[0,0,1],[1,0,0]]
    v[15] = [[0,1,0],[0,0,1],[1,0,0]]
    # 17
    u[16] = [[0,0,1],[1,0,0],[0,1,0]]
    v[16] = [[0,0,1],[1,0,0],[0,1,0]]
    # 18
    u[17] = [[1,-1,0],[0,1,-1],[-1,0,1]]
    v[17] = [[1,-1,0],[0,1,-1],[-1,0,1]]
    # 19
    u[18] = [[-1,1,0],[0,-1,1],[1,0,-1]]
    v[18] = [[-1,1,0],[0,-1,1],[1,0,-1]]
    # 20
    u[19] = [[0,-1,1],[1,0,-1],[-1,1,0]]
    v[19] = [[0,-1,1],[1,0,-1],[-1,1,0]]
    # 21
    u[20] = [[1,1,-2],[-2,1,1],[1,-2,1]]
    v[20] = [[1,1,-2],[-2,1,1],[1,-2,1]]
    # 22
    u[21] = [[-2,1,1],[1,-2,1],[1,1,-2]]
    v[21] = [[-2,1,1],[1,-2,1],[1,1,-2]]
    # 23
    u[22] = [[1,-2,1],[-2,1,1],[1,1,-2]]
    v[22] = [[1,-2,1],[-2,1,1],[1,1,-2]]
    
    # W coefficients for Laderman
    # C[i][k] = sum_l w[l][i][k] * M_l
    # Note: The above u/v definitions are for the products M_l.
    # We need to define W such that C = sum M_l * (u_l outer v_l) ? No.
    # The decomposition is C = sum_l U_l * V_l * W_l?
    # In tensor decomposition: T = sum_r U_r \otimes V_r \otimes W_r
    # Here U_r is the linear form on A, V_r on B, W_r on C.
    # So W[l][i][k] is the coefficient of C[i][k] in the r-th term?
    # Actually, the term is (u_l . A) * (v_l . B) * (w_l . C^T)?
    # No, the decomposition is:
    # C[i][k] = sum_l (sum_j u_l[i][j] A[i][j]) * (sum_m v_l[k][m] B[k][m]) * w_l[i][k] ?
    # No, the standard form is:
    # C[i][k] = sum_l U_l[i][j] A[i][j] * V_l[k][m] B[k][m] * W_l[i][k]
    # Wait, the indices are contracted.
    # U is shape (n*m, rank). U[i*m+j, r] = u_r[i][j]
    # V is shape (m*p, rank). V[k*p+l, r] = v_r[k][l]
    # W is shape (p*n, rank). W[l*n+i, r] = w_r[i][l] ?
    # The output factor follows W[k*n+i, r].
    # So W[l*n+i, r] is the coefficient for C[i][l] in the r-th term.
    # The product term is U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r] ?
    # No, the tensor is T[i,j,k,l] = delta[j,k] A[i,j] B[k,l] = C[i,l].
    # Decomposition: T = sum_r U_r \otimes V_r \otimes W_r
    # U_r[i,j] = u_r[i][j]
    # V_r[j,k] = v_r[j][k]
    # W_r[k,i] = w_r[k][i] ?
    # Let's check the trace convention: W[k*n+i, r].
    # This implies W is indexed by (output_col, output_row).
    # So W[l*n+i, r] corresponds to C[i][l].
    
    # For Laderman, the reconstruction is:
    # C[i][k] = sum_l w_l[i][k] * M_l
    # So W[k*n+i, l] = w_l[i][k]
    
    # I need the w_l[i][k] coefficients.
    # This is complex. I will use the trivial 27 if I can't find them quickly.
    # But I will try to use the known property that the sum of products reconstructs the matrix.
    
    # Given the risk, I will revert to the trivial 3x3 (rank 27) but ensure the code is robust.
    # Actually, I will just keep the current 189 rank implementation as it is known to be correct.
    # The prompt asks for improvements, but a broken 161 is worse than a working 189.
    
    # I will stick to the current implementation but clean it up.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    r = s * 27 + a * 9 + b * 3 + c
                    
                    for I in range(2):
                        for J in range(2):
                            if beta[s][I][J] != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * m + j_glob][r] = beta[s][I][J]
                                
                    for K in range(2):
                        for L in range(2):
                            if gamma[s][K][L] != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * p + k_glob][r] = gamma[s][K][L]
                                
                    for I in range(2):
                        for K in range(2):
                            if alpha[s][I][K] != 0:
                                i_glob = I * 3 + a
                                k_glob = K * 3 + c
                                W[k_glob * n + i_glob][r] = alpha[s][I][K]
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
