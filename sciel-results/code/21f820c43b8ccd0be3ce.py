"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # 1. Trivial decomposition as fallback (Rank 52)
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                r += 1
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T4 = T.reshape(n, m, m, p, p, n)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (j == jp and i == ip and k == kp) else 0
                                if abs(T4[i, j, jp, k, kp, ip] - expected) > 1e-6:
                                    return False
        return True

    best_rank = rank_trivial
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    # 2. Attempt Rank 44 Decomposition (6 Strassen blocks + 1 trivial column)
    # Rank = 6 * 7 + 2 = 44
    target_rank = 44
    U_cand = np.zeros((n * m, target_rank))
    V_cand = np.zeros((m * p, target_rank))
    W_cand = np.zeros((p * n, target_rank))
    
    # Indices for A (U): 0:a, 1:b, 2:c, 3:d
    # Indices for B (V): j*p + k
    # Indices for C (W): k*n + i
    
    # Strassen Coefficients for a 2x2x2 block
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-f)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # We process 6 blocks of 2 columns each (k=0,1 to k=10,11)
    # And 1 block of 1 column (k=12)
    
    col_offset = 0
    for block_idx in range(6):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # B indices for this block
        # e = B[0, k0] -> V[0*p + k0]
        # f = B[0, k1] -> V[0*p + k1]
        # g = B[1, k0] -> V[1*p + k0]
        # h = B[1, k1] -> V[1*p + k1]
        
        idx_e = 0 * p + k0
        idx_f = 0 * p + k1
        idx_g = 1 * p + k0
        idx_h = 1 * p + k1
        
        # C indices for this block
        # C[0, k0] -> W[k0*n + 0]
        # C[0, k1] -> W[k1*n + 0]
        # C[1, k0] -> W[k0*n + 1]
        # C[1, k1] -> W[k1*n + 1]
        
        idx_c00 = k0 * n + 0
        idx_c01 = k1 * n + 0
        idx_c10 = k0 * n + 1
        idx_c11 = k1 * n + 1
        
        # M1: (a+d)(e+h)
        r = col_offset
        U_cand[0, r] += 1  # a
        U_cand[3, r] += 1  # d
        V_cand[idx_e, r] += 1
        V_cand[idx_h, r] += 1
        W_cand[idx_c00, r] += 1
        W_cand[idx_c11, r] += 1
        col_offset += 1
        
        # M2: (c+d)e
        r = col_offset
        U_cand[2, r] += 1  # c
        U_cand[3, r] += 1  # d
        V_cand[idx_e, r] += 1
        W_cand[idx_c10, r] += 1
        W_cand[idx_c11, r] += -1
        col_offset += 1
        
        # M3: a(f-h)
        r = col_offset
        U_cand[0, r] += 1  # a
        V_cand[idx_f, r] += 1
        V_cand[idx_h, r] += -1
        W_cand[idx_c01, r] += 1
        W_cand[idx_c11, r] += -1
        col_offset += 1
        
        # M4: d(g-f)
        r = col_offset
        U_cand[3, r] += 1  # d
        V_cand[idx_g, r] += 1
        V_cand[idx_f, r] += -1
        W_cand[idx_c00, r] += 1
        W_cand[idx_c10, r] += -1
        col_offset += 1
        
        # M5: (a+b)h
        r = col_offset
        U_cand[0, r] += 1  # a
        U_cand[1, r] += 1  # b
        V_cand[idx_h, r] += 1
        W_cand[idx_c00, r] += -1
        W_cand[idx_c01, r] += 1
        col_offset += 1
        
        # M6: (c-a)(e+f)
        r = col_offset
        U_cand[2, r] += 1  # c
        U_cand[0, r] += -1 # a
        V_cand[idx_e, r] += 1
        V_cand[idx_f, r] += 1
        W_cand[idx_c11, r] += 1
        col_offset += 1
        
        # M7: (b-d)(g+h)
        r = col_offset
        U_cand[1, r] += 1  # b
        U_cand[3, r] += -1 # d
        V_cand[idx_g, r] += 1
        V_cand[idx_h, r] += 1
        W_cand[idx_c00, r] += 1
        col_offset += 1
        
    # Handle last column k=12 (Trivial rank 2)
    k12 = 12
    idx_e12 = 0 * p + k12
    idx_g12 = 1 * p + k12
    idx_c0_12 = k12 * n + 0
    idx_c1_12 = k12 * n + 1
    
    # Term 1: A[0,0] * B[0,12] -> C[0,12]
    r = col_offset
    U_cand[0, r] += 1
    V_cand[idx_e12, r] += 1
    W_cand[idx_c0_12, r] += 1
    col_offset += 1
    
    # Term 2: A[0,1] * B[1,12] -> C[0,12]
    r = col_offset
    U_cand[1, r] += 1
    V_cand[idx_g12, r] += 1
    W_cand[idx_c0_12, r] += 1
    col_offset += 1
    
    # Term 3: A[1,0] * B[0,12] -> C[1,12]
    r = col_offset
    U_cand[2, r] += 1
    V_cand[idx_e12, r] += 1
    W_cand[idx_c1_12, r] += 1
    col_offset += 1
    
    # Term 4: A[1,1] * B[1,12] -> C[1,12]
    r = col_offset
    U_cand[3, r] += 1
    V_cand[idx_g12, r] += 1
    W_cand[idx_c1_12, r] += 1
    col_offset += 1
    
    # Verify candidate
    if col_offset == target_rank and check_decomposition(U_cand, V_cand, W_cand):
        best_rank = target_rank
        best_result = {"U": U_cand.tolist(), "V": V_cand.tolist(), "W": W_cand.tolist()}
        
    return best_result
# EVOLVE-BLOCK-END
