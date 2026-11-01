import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    size_a = n * m
    size_b = m * p
    size_c = p * n

    # Build the true tensor T[a, b, c]
    T_true = np.zeros((size_a, size_b, size_c), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1

    # Strassen factors for 2x2x2 (rank 7)
    U_strassen = [
        [1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0]
    ]
    V_strassen = [
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, -1, 0, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    W_strassen = [
        [1, -1, 1, 0, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ]

    U_s = np.array(U_strassen)
    V_s = np.array(V_strassen)
    W_s = np.array(W_strassen)

    # Verify Strassen on 2x2x2
    T_s_recon = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
    T_s_true = np.zeros((4, 4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_s_true[i*2+j, j*2+k, k*2+i] = 1

    strassen_valid = np.array_equal(T_s_recon, T_s_true)

    # Start with Strassen block decomposition (rank 49)
    rank = 7 * (p // 2)
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)

    for b in range(p // 2):
        b_idx_map = [2*b, 2*b+1, p+2*b, p+2*b+1]
        c_idx_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for r in range(7):
            global_col = b * 7 + r
            for i_local in range(4):
                U[i_local, global_col] = U_strassen[i_local][r]
            for j_local in range(4):
                V[b_idx_map[j_local], global_col] = V_strassen[j_local][r]
            for k_local in range(4):
                W[c_idx_map[k_local], global_col] = W_strassen[k_local][r]

    # Verify full decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)

    if not np.array_equal(T_recon, T_true):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding linear dependencies among columns
    # Each column r has support: A indices {0,1,2,3}, B indices {b_idx_map[0..3]}, C indices {c_idx_map[0..3]}
    # We can try to find a linear combination of columns that gives zero
    # This is equivalent to finding the null space of the tensor formed by the columns

    # Reshape: each column r defines a tensor T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # We want to find coefficients alpha_r such that sum_r alpha_r * T_r = 0
    # This is a linear system: for each (a,b,c), sum_r alpha_r * U[a,r]*V[b,r]*W[c,r] = 0

    # Build the constraint matrix M where M[(a,b,c), r] = U[a,r]*V[b,r]*W[c,r]
    # But this is huge. Instead, we can use the structure.

    # Alternative approach: try to find if any column is a linear combination of others
    # by checking if the tensor T_r is in the span of other tensors.

    # For efficiency, let's try a different approach:
    # Use the fact that the problem has a regular structure (14 = 7*2)
    # and try to find a smaller decomposition.

    # Known results: for 2x2x(2k), the rank is at most 7k (Strassen)
    # For 2x2x14 = 2x2x(2*7), rank <= 49
    # Can we do better? Let's try to find a rank 48 decomposition.

    # Strategy: try to merge two Strassen blocks into a rank-13 decomposition
    # instead of two rank-7 = 14.

    # For 2x2x4, the rank is known to be 14 (two Strassen blocks)
    # Actually, for 2x2x4, the rank might be less than 14.
    # Let's try to find a rank-13 decomposition for 2x2x4.

    # If we can find a rank-13 decomposition for 2x2x4, then for 2x2x14 = 2x2x4 + 2x2x4 + 2x2x4 + 2x2x2
    # we get rank 13*3 + 7 = 46.

    # Let's try to find a rank-13 decomposition for 2x2x4.
    # This is equivalent to finding a rank-13 decomposition of the 2x2x4 tensor.

    # The 2x2x4 tensor has 2*2*4 = 16 non-zero entries.
    # We need to find U (4x13), V (8x13), W (8x13) such that the tensor is decomposed.

    # This is a hard problem. Let's try a different approach.

    # Use the fact that the problem is linear and try to find a smaller decomposition
    # by solving a linear system.

    # For each rank r from 48 down to 40, try to find a decomposition.
    # This is computationally expensive, so we'll try a few values.

    # Actually, let's try a more systematic approach.
    # The tensor for 2x2x14 can be seen as a block matrix.
    # A is 2x2, B is 2x14, C is 14x2.

    # The tensor T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.

    # Let's try to use the structure to find a smaller decomposition.

    # For now, let's return the Strassen decomposition with rank 49.
    # This is a valid decomposition and is likely close to optimal.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
