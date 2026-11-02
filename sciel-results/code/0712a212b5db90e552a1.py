import numpy as np
import time

n, m, p = 2, 4, 6
RANK_SCHOOLBOOK = n * m * p

def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    deadline = time.time() + 140.0
    
    sU, sV, sW = _strassen_2x2x2()
    
    nj_blocks = m // 2
    nk_blocks = p // 2
    rank = nj_blocks * nk_blocks * 7
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for jb in range(nj_blocks):
        for kb in range(nk_blocks):
            for c in range(7):
                for i in range(n):
                    for jl in range(2):
                        j_global = jb * 2 + jl
                        U[i * m + j_global, col] = sU[i * 2 + jl, c]
                
                for jl in range(2):
                    for kl in range(2):
                        j_global = jb * 2 + jl
                        k_global = kb * 2 + kl
                        V[j_global * p + k_global, col] = sV[jl * 2 + kl, c]
                
                for kl in range(2):
                    for i in range(n):
                        k_global = kb * 2 + kl
                        W[k_global * n + i, col] = sW[kl * 2 + i, c]
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if col < rank:
        # Fallback to schoolbook if timeout (unlikely)
        U = np.zeros((n * m, RANK_SCHOOLBOOK), dtype=int)
        V = np.zeros((m * p, RANK_SCHOOLBOOK), dtype=int)
        W = np.zeros((p * n, RANK_SCHOOLBOOK), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
