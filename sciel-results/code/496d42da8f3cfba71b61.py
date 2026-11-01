import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The tensor decomposes into 7 independent 2x2 * 2x2 sub-problems (sharing A).
    Each sub-problem has rank 7 (Strassen's algorithm).
    Total rank = 7 * 7 = 49, which is minimal.
    """
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    num_blocks = p // 2  # 7 blocks of 2 columns each

    # Strassen terms for 2x2 * 2x2
    # Each term: (a_indices, a_vals, b_indices, b_vals, c_indices, c_vals)
    # A index: i*m+j (i=row, j=col of A)
    # B index: j*2+k (j=row, k=col of B within 2x2 block)
    # C index: i*2+k (i=row, k=col of C within 2x2 block)
    strassen_terms = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    for t in range(num_blocks):
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            a_idx_list, a_val_list, b_idx_list, b_val_list, c_idx_list, c_val_list = strassen_terms[s]

            for a_idx, coeff in zip(a_idx_list, a_val_list):
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            for b_idx, coeff in zip(b_idx_list, b_val_list):
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            for c_idx, coeff in zip(c_idx_list, c_val_list):
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Deadline guard
    if time.time() > deadline - 5.0:
        # Should not happen for this fixed-size problem, but safety
        pass

    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
