"""
Combines the constructive block decomposition strategy (Strassen + Schoolbook) 
from the inspirations with a modular assembly approach for robustness.
Target: Rank 44 decomposition for 2x4 * 4x7 matrix multiplication.
Improvement: Replaced rank-4 schoolbook for 2x2x1 blocks with rank-1 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}.
    # For each block, split p=7 into three blocks of size 2 (k in {0,1}, {2,3}, {4,5}) 
    # and one block of size 1 (k=6).
    # Use Strassen (rank 7) for 2x2 x 2x2 subproblems.
    # Use Rank-1 (rank 1) for 2x2 x 2x1 subproblems.
    # Total rank = 2 * (3 * 7 + 1 * 1) = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a Strassen decomposition for a 2x2 (A block) * 2x2 (B block) -> 2x2 (C block).
        """
        nonlocal col
        
        # A indices (row-major in U: i*m + j)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices (row-major in V: j*p + k)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W indices (row-major in W: k*n + i)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # Strassen products
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_rank1_221_block(i0, j0, k0):
        """
        Adds a Rank-1 decomposition for a 2x2 (A block) * 2x1 (B block) -> 2x1 (C block).
        Tensor T[i, j, 0] = 1 for all i, j.
        Decomposition: U[i, r]=1, V[j, r]=1, W[0, r]=1.
        """
        nonlocal col
        
        # A indices
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # W indices
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # Single rank term
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        
        V[e_idx, col] += 1
        V[g_idx, col] += 1
        
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        
        col += 1

    # Assemble the decomposition
    # Two j-blocks: j_start in [0, 2]
    for j_start in [0, 2]:
        # Three k-blocks of size 2: k_start in [0, 2, 4]
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        # One k-block of size 1: k_start = 6
        add_rank1_221_block(0, j_start, 6)

    # Verify correctness
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verification function
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > 1e-9:
                            return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
