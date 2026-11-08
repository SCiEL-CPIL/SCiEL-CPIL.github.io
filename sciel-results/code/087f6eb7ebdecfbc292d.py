import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Define the 7 Strassen terms for 2x2x2
    # U2, V2, W2 are 4x7 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, -1, -1, -1, 1, 0, 0]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # The 4x4x4 decomposition is the Kronecker product of the 2x2x2 decompositions
    # Rank = 7 * 7 = 49
    
    # Pre-compute the indices and coefficients for the 49 terms
    # Term r corresponds to (r1, r2) where r1, r2 in 0..6
    # U_4[idx, r] = U2[i1, r1] * U2[i2, r2]
    
    terms = []
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            u_term = np.outer(U2[:, r1], U2[:, r2])
            v_term = np.outer(V2[:, r1], V2[:, r2])
            w_term = np.outer(W2[:, r1], W2[:, r2])
            terms.append((r, u_term.flatten(), v_term.flatten(), w_term.flatten()))

    # Fill U, V, W for the first half (k=0..3)
    for r, u_flat, v_flat, w_flat in terms:
        U[:, r] = u_flat
        # V index mapping: (j, k) -> j*8 + k. Here k is 0..3.
        # v_flat is indexed by (j1, j2, k1, k2) -> (j1*2+j2, k1*2+k2)
        # So v_flat index = (j1*2+j2)*4 + (k1*2+k2)
        # We need to map this to j*8 + k.
        # j = j1*2 + j2, k = k1*2 + k2.
        # target index = j*8 + k = (j1*2+j2)*8 + (k1*2+k2)
        # source index = (j1*2+j2)*4 + (k1*2+k2)
        
        v_mapped = np.zeros(32)
        for idx in range(16):
            j_local = idx // 4
            k_local = idx % 4
            v_global = j_local * 8 + k_local
            v_mapped[v_global] = v_flat[idx]
        V[:, r] = v_mapped

        # W index mapping: (k, i) -> k*4 + i.
        # w_flat is indexed by (k1, k2, i1, i2) -> (k1*2+k2)*4 + (i1*2+i2)
        # This is already in the form (k_local)*4 + i_local.
        # Since k_local is 0..3, this fits directly into the first half of W (indices 0..15).
        W[:16, r] = w_flat

    # Fill U, V, W for the second half (k=4..7)
    for r, u_flat, v_flat, w_flat in terms:
        r2 = 49 + r
        U[:, r2] = u_flat
        
        # V mapping: k is now 4..7. k = 4 + k_local.
        # target index = j*8 + (4 + k_local)
        v_mapped = np.zeros(32)
        for idx in range(16):
            j_local = idx // 4
            k_local = idx % 4
            v_global = j_local * 8 + (4 + k_local)
            v_mapped[v_global] = v_flat[idx]
        V[:, r2] = v_mapped

        # W mapping: k is 4..7. k = 4 + k_local.
        # target index = (4 + k_local)*4 + i_local = 16 + k_local*4 + i_local
        W[16:32, r2] = w_flat

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
