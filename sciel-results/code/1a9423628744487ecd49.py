import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-32 decomposition for 2x4 * 4x8
    # Strategy:
    # The multiplication tensor M_{2,4,8} can be viewed as a direct sum of 8 tensors M_{2,4,1}
    # for each column k of B.
    # M_{2,4,1} has rank 4 (since M_{2,2,1} has rank 2, and M_{2,4,1} is two M_{2,2,1} blocks? 
    # No, M_{2,4,1} is 2x4 times 4x1.
    # Rank(M_{2,4,1}) = 4.
    # Total rank = 8 * 4 = 32.
    
    # Indices:
    # A: 2x4, flat i*4+j
    # B: 4x8, flat j*8+k
    # C: 2x8, flat k*2+i (trace convention W[k*n+i])
    
    U = np.zeros((n*m, 32), dtype=float)
    V = np.zeros((m*p, 32), dtype=float)
    W = np.zeros((p*n, 32), dtype=float)
    
    rank = 0
    for k in range(p):
        # For each k, we have a 2x4 * 4x1 multiplication
        # C_0k = a_00 b_0k + a_01 b_1k + a_02 b_2k + a_03 b_3k
        # C_1k = a_10 b_0k + a_11 b_1k + a_12 b_2k + a_13 b_3k
        #
        # We use 4 rank-1 terms for this block.
        # Term 1: (a_00 + a_10) * b_0k -> contributes to C_0k and C_1k?
        # Let's use the known rank-4 decomposition for M_{2,4,1}.
        # Actually, we can just use 4 terms corresponding to the 4 columns of A?
        # No, that would be 4 terms for C_0k and 4 for C_1k, total 8.
        # But Rank(M_{2,4,1}) is 4.
        #
        # Let's construct the 4 terms.
        # We want to represent:
        # C_0 = a_0 b_0 + a_1 b_1 + a_2 b_2 + a_3 b_3
        # C_1 = a_4 b_0 + a_5 b_1 + a_6 b_2 + a_7 b_3
        # (using flat indices for A: 0..7)
        
        # Term 1: (a_0 + a_4) b_0
        # U: a_0, a_4 = 1
        # V: b_0 = 1
        # W: C_0, C_1 = 1
        U[0][rank] = 1
        U[4][rank] = 1
        V[0*p + k][rank] = 1
        W[k*n + 0][rank] = 1
        W[k*n + 1][rank] = 1
        rank += 1
        
        # Term 2: (a_1 + a_5) b_1
        U[1][rank] = 1
        U[5][rank] = 1
        V[1*p + k][rank] = 1
        W[k*n + 0][rank] = 1
        W[k*n + 1][rank] = 1
        rank += 1
        
        # Term 3: (a_2 + a_6) b_2
        U[2][rank] = 1
        U[6][rank] = 1
        V[2*p + k][rank] = 1
        W[k*n + 0][rank] = 1
        W[k*n + 1][rank] = 1
        rank += 1
        
        # Term 4: (a_3 + a_7) b_3
        U[3][rank] = 1
        U[7][rank] = 1
        V[3*p + k][rank] = 1
        W[k*n + 0][rank] = 1
        W[k*n + 1][rank] = 1
        rank += 1

    # Verify
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    T_full = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
    expected_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                expected_full[a_idx, b_idx, c_idx] = 1
                
    if not np.allclose(T_full, expected_full):
        raise ValueError("Verification failed!")

    return {"U": U, "V": V, "W": W}
