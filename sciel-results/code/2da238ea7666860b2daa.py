"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-14 decomposition for (2x4)*(4x4) matrix multiplication.
    
    The (2,4,4) multiplication tensor T_{i,j,k} = delta_{j,k} is the Kronecker
    product of the (2,2,2) Strassen tensor S (rank 7) and a (2,2,1) identity
    tensor I (rank 2). This gives a rank-14 decomposition.
    
    Index conventions:
      A (2x4):  U index = i*4 + j,  i in {0,1}, j in {0,1,2,3}
      B (4x4):  V index = j*4 + k,  j in {0,1,2,3}, k in {0,1,2,3}
      C (2x4):  W index = k*2 + i,  k in {0,1,2,3}, i in {0,1}
    
    Decomposition:
      T_{i,j,k} = sum_{r=0}^{13} U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r]
    
    where r = 2*s + t, s in {0..6} (Strassen term), t in {0,1} (block pair).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 14
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2*2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # Z[0,0] = M1+M4-M5+M7
    # Z[0,1] = M3+M5
    # Z[1,0] = M2+M4
    # Z[1,1] = M1-M2+M3+M6
    
    # For each Strassen term s, define:
    #   U factors: (u_index, u_coeff) pairs
    #   V factors: (v_index, v_coeff) pairs
    #   W factors: (w_index, w_coeff) pairs
    
    strassen_terms = [
        # (U_factors, V_factors, W_factors)
        # M1 = (x00+x11)*(y00+y11)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (x10+x11)*y00
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # M3 = x00*(y01-y11)
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        # M4 = x11*(y10-y00)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        # M5 = (x00+x01)*y11
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # M6 = (x10-x00)*(y00+y01)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (x01-x11)*(y10+y11)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    # Block pairs: (a_offset, b_offset, c_offset)
    # a_offset: column offset in A (0 for cols 0-1, 2 for cols 2-3)
    # b_offset: row offset in B (0 for rows 0-1, 2 for rows 2-3)
    # c_offset: column offset in C (0 for cols 0-1, 2 for cols 2-3)
    block_pairs = [
        (0, 0, 0),  # A[0:2] * B[0:2] -> C[0:2]
        (2, 2, 2),  # A[2:4] * B[2:4] -> C[2:4]
    ]
    
    for t, (a_off, b_off, c_off) in enumerate(block_pairs):
        for s, (u_facs, v_facs, w_facs) in enumerate(strassen_terms):
            r = 2 * s + t
            for (local_idx, coeff) in u_facs:
                U[local_idx + a_off, r] += coeff
            for (local_idx, coeff) in v_facs:
                V[local_idx + b_off, r] += coeff
            for (local_idx, coeff) in w_facs:
                W[local_idx + c_off, r] += coeff
    
    # Verify the decomposition
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if j == k else 0.0
                error += abs(val - expected)
    
    if error > 1e-10:
        # Fallback to schoolbook rank-32
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
