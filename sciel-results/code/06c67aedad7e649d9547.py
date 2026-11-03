"""Matrix multiplication 3x3 x 3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a rank-34 decomposition for 3x3 * 3x4 matrix multiplication.
    Uses a splitting strategy:
    1. Split output columns k into {0,1} and {2,3}.
    2. Each half is a 3x3x2 problem.
    3. Each 3x3x2 is split into:
       - 2x2x2 (Strassen, rank 7) for i in {0,1}, j in {0,1}
       - 2x1x2 (rank 4) for i in {0,1}, j=2
       - 1x2x2 (rank 4) for i=2, j in {0,1}
       - 1x1x2 (rank 2) for i=2, j=2
    Total rank = 2 * (7 + 4 + 4 + 2) = 34.
    """
    n, m, p = 3, 3, 4
    rank = 34
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    # Strassen terms for 2x2x2
    # U indices: i*2 + j (local)
    # V indices: j*2 + k (local)
    # W indices: k*2 + i (local)
    
    strassen_U = [
        [(0,1), (3,1)],       # P1: A00+A11
        [(2,1), (3,1)],       # P2: A10+A11
        [(0,1)],              # P3: A00
        [(3,1)],              # P4: A11
        [(0,1), (1,1)],       # P5: A00+A01
        [(2,1), (0,-1)],      # P6: A10-A00
        [(1,1), (3,-1)]       # P7: A01-A11
    ]
    strassen_V = [
        [(0,1), (3,1)],       # P1: B00+B11
        [(0,1)],              # P2: B00
        [(2,1), (3,-1)],      # P3: B10-B11
        [(1,1), (3,1)],       # P4: B01+B11
        [(3,1)],              # P5: B11
        [(0,1), (1,1)],       # P6: B00+B01
        [(2,1), (3,1)]        # P7: B10+B11
    ]
    strassen_W = [
        [(0,1), (3,1)],       # P1: C00+C11
        [(2,1), (3,-1)],      # P2: C10-C11
        [(1,1), (3,1)],       # P3: C01+C11
        [(0,1), (2,1)],       # P4: C00+C10
        [(0,1), (1,-1)],      # P5: C00-C01
        [(2,1)],              # P6: C10
        [(1,1), (3,-1)]       # P7: C01-C11
    ]

    for k_offset in [0, 2]:
        # Block 1: i in {0,1}, j in {0,1}, k in {k_offset, k_offset+1}
        # Strassen
        for r in range(7):
            # U
            for (idx, val) in strassen_U[r]:
                i, j = idx // 2, idx % 2
                U[i*m + j, col] = val
            # V
            for (idx, val) in strassen_V[r]:
                j, k_local = idx // 2, idx % 2
                V[j*p + (k_local + k_offset), col] = val
            # W
            for (idx, val) in strassen_W[r]:
                k_local, i = idx // 2, idx % 2
                W[(k_local + k_offset)*n + i, col] = val
            col += 1

        # Block 2: i in {0,1}, j=2, k in {k_offset, k_offset+1}
        # Terms: A[i,2] * B[2, k]
        for i in [0, 1]:
            for k_local in [0, 1]:
                k = k_local + k_offset
                U[i*m + 2, col] = 1
                V[2*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

        # Block 3: i=2, j in {0,1}, k in {k_offset, k_offset+1}
        # Terms: A[2, j] * B[j, k]
        for j in [0, 1]:
            for k_local in [0, 1]:
                k = k_local + k_offset
                U[2*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + 2, col] = 1
                col += 1

        # Block 4: i=2, j=2, k in {k_offset, k_offset+1}
        # Terms: A[2, 2] * B[2, k]
        for k_local in [0, 1]:
            k = k_local + k_offset
            U[2*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
