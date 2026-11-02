import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    nr, mr, pr = n * m, m * p, p * n  # 4, 16, 16
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # ------------------------------------------------------------------
    # Rank-28 construction using Strassen's algorithm for each 2x2x2 block.
    # M_{2,2,8} is the direct sum of four M_{2,2,2} tensors.
    # Rank(M_{2,2,2}) = 7 (Strassen).
    # Total Rank = 4 * 7 = 28.
    # ------------------------------------------------------------------
    
    rank = 28
    U = np.zeros((nr, rank), dtype=float)
    V = np.zeros((mr, rank), dtype=float)
    W = np.zeros((pr, rank), dtype=float)
    
    # Strassen's 7 products for 2x2 matrix multiplication.
    # A_sub is 2x2, B_sub is 2x2, C_sub = A_sub * B_sub.
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Coefficients for U (A-side), V (B-side), W (C-side)
    # Keys are (row, col) for U and V, (col, row) for W?
    # W index in tensor T is c = k*n + i.
    # In the block, c_sub = k_sub * n + i.
    # So W key is (i, k_sub) mapping to index k_sub * n + i.
    
    strassen_products = [
        # M1
        {'U': {(0, 0): 1, (1, 1): 1}, 
         'V': {(0, 0): 1, (1, 1): 1}, 
         'W': {(0, 0): 1, (1, 1): 1}},
        # M2
        {'U': {(1, 0): 1, (1, 1): 1}, 
         'V': {(0, 0): 1}, 
         'W': {(1, 0): 1, (1, 1): -1}},
        # M3
        {'U': {(0, 0): 1}, 
         'V': {(0, 1): 1, (1, 1): -1}, 
         'W': {(0, 1): 1, (1, 1): 1}},
        # M4
        {'U': {(1, 1): 1}, 
         'V': {(1, 0): 1, (0, 0): -1}, 
         'W': {(0, 0): 1, (1, 0): 1}},
        # M5
        {'U': {(0, 0): 1, (0, 1): 1}, 
         'V': {(1, 1): 1}, 
         'W': {(0, 0): -1, (0, 1): 1}},
        # M6
        {'U': {(1, 0): 1, (0, 0): -1}, 
         'V': {(0, 0): 1, (0, 1): 1}, 
         'W': {(1, 1): 1}},
        # M7
        {'U': {(0, 1): 1, (1, 1): -1}, 
         'V': {(1, 0): 1, (1, 1): 1}, 
         'W': {(0, 0): 1}},
    ]

    for q in range(p // 2):
        k1 = 2 * q
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            # U index: i * m + j
            for (i, j), coeff in prod['U'].items():
                U[i * m + j, t] = coeff
            # V index: j * p + (k1 + k_sub)
            for (j, k_sub), coeff in prod['V'].items():
                V[j * p + (k1 + k_sub), t] = coeff
            # W index: (k1 + k_sub) * n + i
            # prod['W'] keys are (i, k_sub)
            for (i, k_sub), coeff in prod['W'].items():
                W[(k1 + k_sub) * n + i, t] = coeff

    # Verify the decomposition
    # T[a,b,c] should be 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nr, mr, pr), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Check time before expensive verification (though it's cheap here)
    if time.time() - start_time > deadline - safety_margin:
        # Return current best (which is rank 28)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    Tc = np.einsum('at,bt,ct->abc', U, V, W)
    err = float(np.max(np.abs(Tc - T)))
    
    if err > 1e-8:
        # Fallback to schoolbook rank 32 if Strassen implementation failed
        rank_fallback = n * m * p
        U_f = np.zeros((nr, rank_fallback), dtype=float)
        V_f = np.zeros((mr, rank_fallback), dtype=float)
        W_f = np.zeros((pr, rank_fallback), dtype=float)
        for idx in range(rank_fallback):
            i = idx // (m * p)
            j = (idx // p) % m
            k = idx % p
            U_f[i * m + j, idx] = 1.0
            V_f[j * p + k, idx] = 1.0
            W_f[k * n + i, idx] = 1.0
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
