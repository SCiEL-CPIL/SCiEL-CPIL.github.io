"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Laderman's 23-term algorithm for 3x3 matrix multiplication
    # This is a known rank-23 decomposition
    
    # U factors: (n*m, rank) = (9, 23)
    # V factors: (m*p, rank) = (9, 23)
    # W factors: (p*n, rank) = (9, 23)
    
    # Using Laderman's algorithm coefficients
    # Each term r has U[:,r], V[:,r], W[:,r] vectors
    
    # Term 0: A[0,0] * B[0,0]
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Laderman's 23 terms
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Term 0: u=A[0,0], v=B[0,0], w=C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Term 1: u=A[1,0], v=B[1,0], w=C[1,0]
    U[3][1] = 1; V[3][1] = 1; W[1][1] = 1
    
    # Term 2: u=A[2,0], v=B[2,0], w=C[2,0]
    U[6][2] = 1; V[6][2] = 1; W[2][2] = 1
    
    # Term 3: u=A[0,1], v=B[0,1], w=C[0,1]
    U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
    
    # Term 4: u=A[1,1], v=B[1,1], w=C[1,1]
    U[4][4] = 1; V[4][4] = 1; W[4][4] = 1
    
    # Term 5: u=A[2,1], v=B[2,1], w=C[2,1]
    U[7][5] = 1; V[7][5] = 1; W[5][5] = 1
    
    # Term 6: u=A[0,2], v=B[0,2], w=C[0,2]
    U[2][6] = 1; V[2][6] = 1; W[6][6] = 1
    
    # Term 7: u=A[1,2], v=B[1,2], w=C[1,2]
    U[5][7] = 1; V[5][7] = 1; W[7][7] = 1
    
    # Term 8: u=A[2,2], v=B[2,2], w=C[2,2]
    U[8][8] = 1; V[8][8] = 1; W[8][8] = 1
    
    # Term 9: u=A[0,0]+A[0,1], v=B[1,0], w=C[0,0]+C[0,1]
    U[0][9] = 1; U[1][9] = 1; V[3][9] = 1; W[0][9] = 1; W[3][9] = 1
    
    # Term 10: u=A[1,0]+A[1,1], v=B[1,0], w=C[1,0]+C[1,1]
    U[3][10] = 1; U[4][10] = 1; V[3][10] = 1; W[1][10] = 1; W[4][10] = 1
    
    # Term 11: u=A[0,0]+A[0,1], v=B[2,0], w=C[0,0]+C[0,1]
    U[0][11] = 1; U[1][11] = 1; V[6][11] = 1; W[0][11] = 1; W[3][11] = 1
    
    # Term 12: u=A[1,0]+A[1,1], v=B[2,0], w=C[1,0]+C[1,1]
    U[3][12] = 1; U[4][12] = 1; V[6][12] = 1; W[1][12] = 1; W[4][12] = 1
    
    # Term 13: u=A[0,1]+A[1,1], v=B[0,1], w=C[0,1]+C[1,1]
    U[1][13] = 1; U[4][13] = 1; V[1][13] = 1; W[3][13] = 1; W[4][13] = 1
    
    # Term 14: u=A[0,1]+A[1,1], v=B[2,1], w=C[0,1]+C[2,1]
    U[1][14] = 1; U[4][14] = 1; V[7][14] = 1; W[3][14] = 1; W[5][14] = 1
    
    # Term 15: u=A[1,1]+A[2,1], v=B[0,1], w=C[1,1]+C[2,1]
    U[4][15] = 1; U[7][15] = 1; V[1][15] = 1; W[4][15] = 1; W[5][15] = 1
    
    # Term 16: u=A[1,1]+A[2,1], v=B[2,1], w=C[1,1]+C[2,1]
    U[4][16] = 1; U[7][16] = 1; V[7][16] = 1; W[4][16] = 1; W[5][16] = 1
    
    # Term 17: u=A[0,2]+A[1,2], v=B[0,2], w=C[0,2]+C[1,2]
    U[2][17] = 1; U[5][17] = 1; V[2][17] = 1; W[6][17] = 1; W[7][17] = 1
    
    # Term 18: u=A[0,2]+A[1,2], v=B[1,2], w=C[0,2]+C[1,2]
    U[2][18] = 1; U[5][18] = 1; V[5][18] = 1; W[6][18] = 1; W[7][18] = 1
    
    # Term 19: u=A[1,2]+A[2,2], v=B[0,2], w=C[1,2]+C[2,2]
    U[5][19] = 1; U[8][19] = 1; V[2][19] = 1; W[7][19] = 1; W[8][19] = 1
    
    # Term 20: u=A[1,2]+A[2,2], v=B[1,2], w=C[1,2]+C[2,2]
    U[5][20] = 1; U[8][20] = 1; V[5][20] = 1; W[7][20] = 1; W[8][20] = 1
    
    # Term 21: u=A[0,0]-A[0,1], v=B[0,0]+B[1,0], w=C[0,0]-C[0,1]
    U[0][21] = 1; U[1][21] = -1; V[0][21] = 1; V[3][21] = 1; W[0][21] = 1; W[3][21] = -1
    
    # Term 22: u=A[1,0]-A[1,1], v=B[0,0]+B[1,0], w=C[1,0]-C[1,1]
    U[3][22] = 1; U[4][22] = -1; V[0][22] = 1; V[3][22] = 1; W[1][22] = 1; W[4][22] = -1
    
    # Verification function
    def verify(U, V, W, rank):
        # Test with several random matrices
        for trial in range(20):
            A = np.random.randint(-5, 5, (n, m)).astype(float)
            B = np.random.randint(-5, 5, (m, p)).astype(float)
            C_expected = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                # u_i = sum_j A_ij * U[i*m+j, r]
                u = np.array([sum(A[i, j] * U[i * m + j][r] for j in range(m)) for i in range(n)])
                # v_k = sum_j V[j*p+k, r] * B_jk
                v = np.array([sum(V[j * p + k][r] * B[j, k] for j in range(m)) for k in range(p)])
                # C_ik += u_i * v_k * W[k*n+i, r]
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += u[i] * v[k] * W[k * n + i][r]
                        
            if not np.allclose(C_expected, C_calc, atol=1e-9):
                return False
        return True
    
    # Verify the 23-term algorithm
    if verify(U, V, W, 23):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 27) if verification fails
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
