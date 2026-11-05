import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (col 0) * B_1 (row 0)
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
    # Part 2: A_2 (cols 1,2) * B_2 (rows 1,2) split into 3 blocks of 2x2
    for x in range(3):
        c_start = 2 * x
        
        U[1][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1; V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        U[4][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1
        W[2*c_start + 1][col] = 1
        W[2*c_start + 3][col] = -1
        col += 1
        
        U[1][col] = 1
        V[6 + c_start + 1][col] = 1
        V[12 + c_start + 1][col] = -1
        W[2*c_start + 2][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        U[5][col] = 1
        V[12 + c_start][col] = 1
        V[6 + c_start][col] = -1
        W[2*c_start][col] = 1
        W[2*c_start + 1][col] = 1
        col += 1
        
        U[1][col] = 1; U[2][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = -1
        W[2*c_start + 2][col] = 1
        col += 1
        
        U[4][col] = 1; U[1][col] = -1
        V[6 + c_start][col] = 1
        V[6 + c_start + 1][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        U[2][col] = 1; U[5][col] = -1
        V[12 + c_start][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        col += 1
    
    # Verify: check that sum_r U[i,r]*V[j,r]*W[k,r] equals the multiplication tensor
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] where:
    # i = iA * m + jA (iA in [0,n), jA in [0,m))
    # j = jB * p + kB (jB in [0,m), kB in [0,p))
    # k = kC * n + iC (kC in [0,p), iC in [0,n))
    # T[i,j,k] = 1 iff jA == jB and iA == iC
    
    for iA in range(n):
        for jA in range(m):
            for jB in range(m):
                for kC in range(p):
                    for iC in range(n):
                        i = iA * m + jA
                        j = jB * p + kC
                        k = kC * n + iC
                        expected = 1.0 if (jA == jB and iA == iC) else 0.0
                        actual = sum(U_np[i, r] * V_np[j, r] * W_np[k, r] for r in range(rank))
                        if abs(actual - expected) > 1e-9:
                            # Fallback: return schoolbook rank 36
                            rank_fb = 36
                            U_fb = [[0] * rank_fb for _ in range(n * m)]
                            V_fb = [[0] * rank_fb for _ in range(m * p)]
                            W_fb = [[0] * rank_fb for _ in range(p * n)]
                            col_fb = 0
                            for iA in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        U_fb[iA * m + j][col_fb] = 1
                                        V_fb[j * p + k][col_fb] = 1
                                        W_fb[k * n + iA][col_fb] = 1
                                        col_fb += 1
                            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
