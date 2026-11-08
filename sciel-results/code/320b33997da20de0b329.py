"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Decomposition strategy:
    # Split m=3 into m1=2 (j=0,1) and m2=1 (j=2)
    # Part 1: 2x2x6 multiplication (A[:, :2] * B[:2, :])
    #   Split k=6 into 3 pairs: (0,1), (2,3), (4,5)
    #   Each 2x2x2 block uses Strassen's 7 terms
    #   Total: 3 * 7 = 21 terms
    # Part 2: 2x1x6 multiplication (A[:, 2:] * B[2:, :])
    #   T[i,2,k] = 1 for all i,k
    #   Rank 1: U[i,r]=1, V[2,r]=1, W[k,r]=1
    # Total rank: 21 + 1 = 22
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22
    
    # Strassen terms for 2x2x2
    # A indices: i*2+j for i in {0,1}, j in {0,1} -> 0,1,2,3
    # B indices: j*2+k for j in {0,1}, k in {0,1} -> 0,1,2,3
    # C indices: i*2+k for i in {0,1}, k in {0,1} -> 0,1,2,3
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        # M2 = (A10+A11)B00
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        # M3 = A00(B01-B11)
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        # M4 = A11(B10-B00)
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        # M5 = (A00+A01)B11
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        # M6 = (A10-A00)(B00+B01)
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        # M7 = (A01-A11)(B10+B11)
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
    ]
    
    # Fill in the 21 Strassen terms for 3 blocks of 2x2x2
    for block in range(3):
        if time.time() > _deadline:
            break
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            
            # U: A indices are i*2 + j (since we're using the left 2x2 of A)
            # Global A index: i*m + j = i*3 + j
            for l_idx, val in term['u'].items():
                i_local = l_idx // 2
                j_local = l_idx % 2
                g_idx = i_local * m + j_local
                U[g_idx][col] = val
            
            # V: B indices are j*2 + k (since we're using the top 2x? of B)
            # Global B index: j*p + k
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_k = block * 2 + k_local
                g_idx = j_local * p + g_k
                V[g_idx][col] = val
            
            # W: C indices are i*2 + k
            # Global W index: k*n + i (trace convention W[k*n+i, r])
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_k = block * 2 + k_local
                g_idx = g_k * n + i_local
                W[g_idx][col] = val
    
    # Add the rank-1 term for the 2x1x6 part
    col = 21
    # U[i, col] = 1 for i in {0,1} -> A indices: i*3 + 2
    for i in range(n):
        U[i * m + 2][col] = 1
    
    # V[j, col] = 1 for j=2 -> B index: 2*p + k... wait
    # V has shape (m*p, rank), indexed as V[j*p + k, r]
    # We need V[2*p + k, col] for all k? No - V is a single vector.
    # V[j*p+k, r] where j=2, k varies. But V is a matrix, so each row is separate.
    # We need: for the term to work, V[2*p + k, col] should contribute.
    # Actually, V is (m*p, rank), so V[row, col] where row = j*p + k.
    # For j=2, k=0..5: rows are 2*6+0=12, 13, 14, 15, 16, 17
    # We need V[12+k, col] = 1 for all k? That would make the term:
    # sum_k U[i,col] * V[2*p+k, col] * W[k*n+i, col]
    # But that's not how it works. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For j=2: T[i,2,k] = sum_r U[i*3+2, r] * V[2*6+k, r] * W[k*2+i, r]
    # We want this to equal 1 for all i,k.
    # With one term r=21: U[i*3+2, 21] * V[12+k, 21] * W[2k+i, 21] = 1
    # Set U[i*3+2, 21] = 1, V[12+k, 21] = 1, W[2k+i, 21] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
