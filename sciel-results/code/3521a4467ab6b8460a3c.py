import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Rank 56 decomposition using 8 independent Strassen blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2 matrix multiplication
    # Each term: U coefficients for A[i,j], V coefficients for B[j,k], W coefficients for C[i,k]
    # W uses trace convention: W[k*n + i, r]
    strassen_terms = [
        # M0 = (A00+A11)(B00+B11)
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        # M1 = (A10+A11)B00
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}},
        # M2 = A00(B01-B11)
        {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}},
        # M3 = A11(B10-B00)
        {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        # M4 = (A00+A01)B11
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}},
        # M5 = (A10-A00)(B00+B01)
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        # M6 = (A01-A11)(B10+B11)
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}},
    ]
    
    col = 0
    for j_high in range(2):  # j = 0,1 or 2,3
        for k_high in range(4):  # k = 0,1 or 2,3 or 4,5 or 6,7
            for t_idx in range(7):
                term = strassen_terms[t_idx]
                
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = k_high * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (k_low, i), coeff in term['W'].items():
                    k = k_high * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                    
                col += 1
    
    # Quick verification using numpy
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        test_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        test_ok = False
                        break
                if not test_ok:
                    break
            if not test_ok:
                break
        
        if not test_ok:
            # Fallback: return naive rank 64
            rank_naive = n * m * p
            U_naive = [[0] * rank_naive for _ in range(n * m)]
            V_naive = [[0] * rank_naive for _ in range(m * p)]
            W_naive = [[0] * rank_naive for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U_naive[i * m + j][r] = 1
                        V_naive[j * p + k][r] = 1
                        W_naive[k * n + i][r] = 1
            return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
