"""
Strategy: Systematic Kronecker Product Decomposition.
Instead of manually mapping Strassen terms to specific sub-blocks of the 2x4x6 matrix,
we treat the problem as finding a set of "tiles" (small decompositions) that cover the 
index space. We use the known rank-7 Strassen decomposition for 2x2x2 as our base tile.
We then explore different tilings of the 4x6 inner/output space using 2x2 blocks.
This is a structural change: from manual coordinate mapping to automated block tiling.
"""

DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Base Strassen 2x2x2 decomposition
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # U, V, W are 4x7 matrices (flattened indices)
    
    # Strassen terms defined as (U_dict, V_dict, W_dict)
    # Keys are (local_i, local_j), (local_j, local_k), (local_k, local_i)
    strassen_terms = [
        # r=0: M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # r=1: M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
        # r=2: M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}),
        # r=3: M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # r=4: M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # r=5: M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # r=6: M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    rank_base = 7
    n_base = 2
    m_base = 2
    p_base = 2
    
    # We need to tile the 4x6 space (j,k) with 2x2 blocks.
    # j ranges 0..3, k ranges 0..5.
    # Possible tilings:
    # 1. 2x3 grid of 2x2 blocks (standard)
    # 2. Other tilings? Since dimensions are multiples of 2, standard tiling is natural.
    # Let's generate the decomposition for the 2x3 tiling.
    
    # Global dimensions
    N, M, P = n, m, p
    
    # Number of blocks
    m_blocks = M // m_base # 2
    p_blocks = P // p_base # 3
    
    total_rank = m_blocks * p_blocks * rank_base # 42
    
    # Initialize matrices
    U = np.zeros((N * M, total_rank), dtype=float)
    V = np.zeros((M * P, total_rank), dtype=float)
    W = np.zeros((P * N, total_rank), dtype=float)
    
    col_idx = 0
    for jb in range(m_blocks):
        for kb in range(p_blocks):
            for r in range(rank_base):
                U_dict, V_dict, W_dict = strassen_terms[r]
                
                # Fill U
                for (li, lj), val in U_dict.items():
                    gi = li
                    gj = jb * m_base + lj
                    U[gi * M + gj, col_idx] = val
                
                # Fill V
                for (lj, lk), val in V_dict.items():
                    gj = jb * m_base + lj
                    gk = kb * p_base + lk
                    V[gj * P + gk, col_idx] = val
                
                # Fill W
                # Trace convention: W[k*n+i, r]
                # In Strassen term, W is indexed by (local_k, local_i) corresponding to C[local_i, local_k]
                # Wait, the Strassen term W_dict keys are (k_local, i_local) in the local 2x2 space?
                # Let's re-verify the Strassen W indices.
                # C[i,k] = sum_r ... W[k,i,r] ...
                # In the base 2x2x2, W is 4x7.
                # Term 0: C00 += 1, C11 += 1.
                # C00 corresponds to i=0, k=0. W[0*2+0, r] = W[0, r] = 1.
                # C11 corresponds to i=1, k=1. W[1*2+1, r] = W[3, r] = 1.
                # My strassen_terms W_dict: {(0,0):1, (1,1):1}.
                # Key (0,0) means k_local=0, i_local=0. Global k=kb*2+0, i=0.
                # Key (1,1) means k_local=1, i_local=1. Global k=kb*2+1, i=1.
                # This matches the standard Strassen C00, C11 contributions.
                
                for (lk, li), val in W_dict.items():
                    gk = kb * p_base + lk
                    gi = li
                    W[gk * N + gi, col_idx] = val
                
                col_idx += 1
    
    # Verify the decomposition
    # T[i,j,k] = 1 for all i,j,k
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = 0.0
                    u_val = U[i*M + j, :]
                    v_val = V[j*P + k, :]
                    w_val = W[k*N + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = N * M * P
        U_s = np.zeros((N * M, rank))
        V_s = np.zeros((M * P, rank))
        W_s = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U_s[i*M + j, col] = 1
                    V_s[j*P + k, col] = 1
                    W_s[k*N + i, col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
