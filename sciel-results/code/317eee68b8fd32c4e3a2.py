"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Strassen decomposition for 4x4 * 4x6
    # Block structure: A is 2x2 blocks of 2x2, B is 2x2 blocks of 2x3
    # 7 Strassen products, each is a 2x2 * 2x3 multiplication (schoolbook rank 12)
    # Total rank = 7 * 12 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # For each Strassen product s (0..6), and for each (di, dj, dk),
    # we have a column c = s*12 + di*6 + dj*3 + dk
    # U[i*m+j, c] = alpha_s[bi, bj, di, dj] where i=bi*2+di, j=bj*2+dj
    # V[j*p+k, c] = beta_s[bj, bk, dj, dk] where j=bj*2+dj, k=bk*3+dk
    # W[k*n+i, c] = gamma_s[bi, bk] where i=bi*2+di, k=bk*3+dk
    
    # Define the Strassen coefficients
    # alpha_s[bi, bj, di, dj]: coefficient of A[bi*2+di, bj*2+dj] in L_s[di, dj]
    # beta_s[bj, bk, dj, dk]: coefficient of B[bj*2+dj, bk*3+dk] in R_s[dj, dk]
    # gamma_s[bi, bk]: coefficient of M_s in result block (bi, bk)
    
    # s=0 (M1): L=A11+A22, R=B11+B22, contributes to C11(+1), C22(+1)
    # s=1 (M2): L=A21+A22, R=B11, contributes to C21(+1), C22(-1)
    # s=2 (M3): L=A11, R=B12-B22, contributes to C12(+1), C22(+1)
    # s=3 (M4): L=A22, R=B21-B11, contributes to C11(+1), C21(+1)
    # s=4 (M5): L=A11+A12, R=B22, contributes to C11(-1), C12(+1)
    # s=5 (M6): L=A21-A11, R=B11+B12, contributes to C22(+1)
    # s=6 (M7): L=A12-A22, R=B21+B22, contributes to C11(+1)
    
    # Let me define these as dictionaries
    
    # alpha[s][bi][bj] = 2x2 matrix (di, dj) -> coefficient
    alpha = {}
    # s=0: L=A11+A22
    alpha[0] = {(0,0): [[1,1],[1,1]], (1,1): [[1,1],[1,1]]}
    # s=1: L=A21+A22
    alpha[1] = {(1,0): [[1,1],[1,1]], (1,1): [[1,1],[1,1]]}
    # s=2: L=A11
    alpha[2] = {(0,0): [[1,1],[1,1]]}
    # s=3: L=A22
    alpha[3] = {(1,1): [[1,1],[1,1]]}
    # s=4: L=A11+A12
    alpha[4] = {(0,0): [[1,1],[1,1]], (0,1): [[1,1],[1,1]]}
    # s=5: L=A21-A11
    alpha[5] = {(1,0): [[1,1],[1,1]], (0,0): [[-1,-1],[-1,-1]]}
    # s=6: L=A12-A22
    alpha[6] = {(0,1): [[1,1],[1,1]], (1,1): [[-1,-1],[-1,-1]]}
    
    # beta[s][bj][bk] = 2x3 matrix (dj, dk) -> coefficient
    beta = {}
    # s=0: R=B11+B22
    beta[0] = {(0,0): [[1,1,1],[1,1,1]], (1,1): [[1,1,1],[1,1,1]]}
    # s=1: R=B11
    beta[1] = {(0,0): [[1,1,1],[1,1,1]]}
    # s=2: R=B12-B22
    beta[2] = {(0,1): [[1,1,1],[1,1,1]], (1,1): [[-1,-1,-1],[-1,-1,-1]]}
    # s=3: R=B21-B11
    beta[3] = {(1,0): [[1,1,1],[1,1,1]], (0,0): [[-1,-1,-1],[-1,-1,-1]]}
    # s=4: R=B22
    beta[4] = {(1,1): [[1,1,1],[1,1,1]]}
    # s=5: R=B11+B12
    beta[5] = {(0,0): [[1,1,1],[1,1,1]], (0,1): [[1,1,1],[1,1,1]]}
    # s=6: R=B21+B22
    beta[6] = {(1,0): [[1,1,1],[1,1,1]], (1,1): [[1,1,1],[1,1,1]]}
    
    # gamma[s][bi][bk] = coefficient
    gamma = {}
    # s=0 (M1): C11(+1), C22(+1)
    gamma[0] = {(0,0): 1, (1,1): 1}
    # s=1 (M2): C21(+1), C22(-1)
    gamma[1] = {(1,0): 1, (1,1): -1}
    # s=2 (M3): C12(+1), C22(+1)
    gamma[2] = {(0,1): 1, (1,1): 1}
    # s=3 (M4): C11(+1), C21(+1)
    gamma[3] = {(0,0): 1, (1,0): 1}
    # s=4 (M5): C11(-1), C12(+1)
    gamma[4] = {(0,0): -1, (0,1): 1}
    # s=5 (M6): C22(+1)
    gamma[5] = {(1,1): 1}
    # s=6 (M7): C11(+1)
    gamma[6] = {(0,0): 1}
    
    # Now fill in U, V, W
    for s in range(7):
        for di in range(2):
            for dj in range(2):
                for dk in range(3):
                    c = s * 12 + di * 6 + dj * 3 + dk
                    
                    # Fill U: for each (bi, bj) in alpha[s], set U[bi*2+di, bj*2+dj] += alpha[s][bi,bj][di][dj]
                    for (bi, bj), mat in alpha[s].items():
                        val = mat[di][dj]
                        if val != 0:
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][c] += val
                    
                    # Fill V: for each (bj, bk) in beta[s], set V[bj*2+dj, bk*3+dk] += beta[s][bj,bk][dj][dk]
                    for (bj, bk), mat in beta[s].items():
                        val = mat[dj][dk]
                        if val != 0:
                            j = bj * 2 + dj
                            k = bk * 3 + dk
                            V[j * p + k][c] += val
                    
                    # Fill W: for each (bi, bk) in gamma[s], set W[bk*3+dk, bi*2+di] += gamma[s][bi,bk]
                    for (bi, bk), val in gamma[s].items():
                        if val != 0:
                            i = bi * 2 + di
                            k = bk * 3 + dk
                            W[k * n + i][c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
