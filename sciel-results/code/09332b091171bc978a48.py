import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen on 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # Schoolbook for the rest: 75 - 8 + 7 = 74 rank
    
    # Strassen 7 products for 2x2x2:
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2: (A10+A11)B00 -> C10+=1, C11+=-1
    # M3: A00(B10-B11) -> C01+=1, C11+=1
    # M4: A11(B01-B00) -> C00+=1, C10+=1
    # M5: (A00+A01)B11 -> C00+=-1, C01+=1
    # M6: (A10-A00)(B00+B10) -> C11+=1
    # M7: (A01-A11)(B01+B11) -> C00+=1, C11+=-1
    
    strassen_terms = [
        # (U_entries: list of (i,j,coeff), V_entries: list of (j,k,coeff), W_entries: list of (k,i,coeff))
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),                # M2
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1),(1,1,1)]),               # M3
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(0,1,1)]),               # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),               # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]),               # M6
        ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,1),(1,1,-1)]),      # M7
    ]
    
    rank = 74
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    # Add Strassen terms for sub-block i in {0,1}, j in {0,1}, k in {0,1}
    for u_entries, v_entries, w_entries in strassen_terms:
        for (i, j, c) in u_entries:
            U[i * m + j, col] += c
        for (j, k, c) in v_entries:
            V[j * p + k, col] += c
        for (k, i, c) in w_entries:
            W[k * n + i, col] += c
        col += 1
    
    # Add schoolbook terms for all other (i,j,k) not in the Strassen sub-block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_seed = n * m * p
        U_seed = np.zeros((n * m, rank_seed), dtype=object)
        V_seed = np.zeros((m * p, rank_seed), dtype=object)
        W_seed = np.zeros((p * n, rank_seed), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_seed[i * m + j, col] = 1
                    V_seed[j * p + k, col] = 1
                    W_seed[k * n + i, col] = 1
                    col += 1
        return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
