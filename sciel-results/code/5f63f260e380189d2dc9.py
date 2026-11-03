"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    factors = []
    
    def decompose(n, m, p, off_i, off_j, off_k):
        if n == 2 and m == 2 and p == 2:
            U_loc = np.zeros((4, 7))
            V_loc = np.zeros((4, 7))
            W_loc = np.zeros((4, 7))
            
            # Strassen setup
            U_loc[0, 0] = 1; V_loc[2, 0] = 1; V_loc[3, 0] = 1
            U_loc[0, 1] = 1; U_loc[2, 1] = 1; V_loc[0, 1] = 1
            U_loc[2, 2] = 1; U_loc[3, 2] = 1; V_loc[3, 2] = 1
            U_loc[3, 3] = 1; V_loc[0, 3] = 1; V_loc[2, 3] = -1
            U_loc[0, 4] = 1; U_loc[3, 4] = 1; V_loc[0, 4] = 1; V_loc[3, 4] = 1
            U_loc[0, 5] = 1; U_loc[2, 5] = -1; V_loc[1, 5] = 1; V_loc[3, 5] = 1
            U_loc[2, 6] = 1; U_loc[3, 6] = -1; V_loc[0, 6] = 1; V_loc[1, 6] = 1
            
            W_loc[0, 4] = 1; W_loc[0, 3] = 1; W_loc[0, 2] = -1; W_loc[0, 6] = 1
            W_loc[1, 0] = 1; W_loc[1, 2] = 1
            W_loc[2, 1] = 1; W_loc[2, 3] = 1
            W_loc[3, 4] = 1; W_loc[3, 0] = -1; W_loc[3, 1] = 1; W_loc[3, 5] = -1
            
            factors.append((U_loc, V_loc, W_loc, n, m, p, off_i, off_j, off_k))
            
        elif n > 1:
            n1, n2 = 1, n - 1
            decompose(n1, m, p, off_i, off_j, off_k)
            decompose(n2, m, p, off_i + n1, off_j, off_k)
            
        elif m > 1:
            m1, m2 = 2 if m >= 2 else 1, m - (2 if m >= 2 else 1)
            decompose(n, m1, p, off_i, off_j, off_k)
            decompose(n, m2, p, off_i, off_j + m1, off_k)
            
        elif p > 1:
            p1, p2 = 2 if p >= 2 else 1, p - (2 if p >= 2 else 1)
            decompose(n, m, p1, off_i, off_j, off_k)
            decompose(n, m, p2, off_i, off_j, off_k + p1)
            
        else:
            rank = n * m * p
            U_loc = np.zeros((n * m, rank))
            V_loc = np.zeros((m * p, rank))
            W_loc = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_loc[i * m + j, col] = 1
                        V_loc[j * p + k, col] = 1
                        W_loc[k * n + i, col] = 1
                        col += 1
            factors.append((U_loc, V_loc, W_loc, n, m, p, off_i, off_j, off_k))

    decompose(n, m, p, 0, 0, 0)
    
    total_rank = sum(U.shape[1] for U, V, W, *_ in factors)
    U_global = np.zeros((n * m, total_rank))
    V_global = np.zeros((m * p, total_rank))
    W_global = np.zeros((p * n, total_rank))
    
    col = 0
    for U_loc, V_loc, W_loc, ln, lm, lp, off_i, off_j, off_k in factors:
        r = U_loc.shape[1]
        
        for i in range(ln):
            for j in range(lm):
                gi = off_i + i
                gj = off_j + j
                U_global[gi * m + gj, col:col+r] += U_loc[i * lm + j, :]
                
        for j in range(lm):
            for k in range(lp):
                gj = off_j + j
                gk = off_k + k
                V_global[gj * p + gk, col:col+r] += V_loc[j * lp + k, :]
                
        for k in range(lp):
            for i in range(ln):
                gk = off_k + k
                gi = off_i + i
                W_global[gk * n + gi, col:col+r] += W_loc[k * ln + i, :]
                
        col += r
        
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
