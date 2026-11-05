import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook to be safe
    if _verify(U_school, V_school, W_school, n, m, p):
        return {"U": U_school, "V": V_school, "W": W_school}

    # Fallback (should not happen)
    return {"U": U_school, "V": V_school, "W": W_school}

def _verify(U, V, W, n, m, p):
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    R = U_np.shape[1]
    
    T_comp = np.zeros((n*m, m*p, p*n))
    for c in range(R):
        T_comp += np.einsum('a,b,c->abc', U_np[:,c], V_np[:,c], W_np[:,c])
        
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T_comp, T_target, atol=1e-9)
# EVOLVE-BLOCK-END
