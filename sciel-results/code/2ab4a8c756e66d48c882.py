"""
Combines the robust Strassen Kronecker construction from the current program
with the clear structure and verification of the inspirations.
Achieves rank 49 for 4x4 matrix multiplication.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the Kronecker product of the rank-7 Strassen decomposition for 2x2.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build 4x4 via Kronecker product of the 2x2 Strassen factors.
    # np.kron(A, B) where A, B are (4, 7) results in (16, 49).
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # Permutation to fix indexing.
    # np.kron row index R = i_A * 4 + i_B.
    # i_A = i_out * 2 + j_out (linear index in 2x2 block)
    # i_B = i_in * 2 + j_in (linear index in 2x2 inner)
    # Target row index S = i * 4 + j.
    # i = i_out * 2 + i_in
    # j = j_out * 2 + j_in
    # S = (i_out * 2 + i_in) * 4 + (j_out * 2 + j_in)
    # S = i_out * 8 + i_in * 4 + j_out * 2 + j_in
    
    # R = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
    # R = i_out * 8 + j_out * 4 + i_in * 2 + j_in
    
    # We need perm[S] = R.
    perm = np.empty(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        
        R = a_idx * 4 + b_idx
        perm[S] = R

    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]

    # Verify 4x4 correctness
    def verify(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = int(np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :]))
                    if s != 1:
                        return False
        return True

    if not verify(U4_fixed, V4_fixed, W4_fixed):
        # Fallback: schoolbook 4x4 (rank 64)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt rank reduction by removing redundant terms (unlikely to succeed for Strassen Kron)
    # But included for completeness as per current program strategy.
    contributing_terms = set()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = U4_fixed[i*4+j, :]
                v_row = V4_fixed[j*4+k, :]
                w_row = W4_fixed[k*4+i, :]
                prod = u_row * v_row * w_row
                for c in range(49):
                    if prod[c] != 0:
                        contributing_terms.add(c)
    
    keep_terms = sorted(contributing_terms)
    if len(keep_terms) < 49:
        U4_reduced = U4_fixed[:, keep_terms]
        V4_reduced = V4_fixed[:, keep_terms]
        W4_reduced = W4_fixed[:, keep_terms]
        
        if verify(U4_reduced, V4_reduced, W4_reduced):
            return {
                "U": U4_reduced.tolist(),
                "V": V4_reduced.tolist(),
                "W": W4_reduced.tolist()
            }

    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
