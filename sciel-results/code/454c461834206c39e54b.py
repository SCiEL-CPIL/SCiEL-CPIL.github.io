"""Matrix multiplication 2,2,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x8 matrix, following the W[k*n+i, r] convention.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    
    The decomposition satisfies:
      T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strassen decomposition for 2x2 matrices
    # Each term is (u, v, w) where u, v, w are length-4 vectors
    # u indexes A (flattened row-major: [A00, A01, A10, A11])
    # v indexes B (flattened row-major: [B00, B01, B10, B11])
    # w indexes C (flattened row-major: [C00, C01, C10, C11])
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    ]

    # Build rank-28 decomposition using 4 Strassen blocks (one per 2x2 block of B)
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    num_blocks = p // 2  # 4 blocks

    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = strassen_terms[term_idx]

            # Fill U: index is i (0..3) for A[i//2, i%2]
            for i in range(4):
                U[i][col] = u[i]

            # Fill V: B is 2x8, block covers columns 2*block_idx and 2*block_idx+1
            # Global B index: row j (0..1), col k_global
            # Local v index: j*2 + k_local where k_local is 0 or 1
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * p + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]

            # Fill W: C is 8x2 (p x n), index is k_global*n + i
            # The local w vector is for a 2x2 block of C.
            # The block of C is at rows 2*block_idx and 2*block_idx+1, cols 0 and 1.
            # So local index (i, k_local) maps to global index (2*block_idx + k_local)*n + i.
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * n + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
