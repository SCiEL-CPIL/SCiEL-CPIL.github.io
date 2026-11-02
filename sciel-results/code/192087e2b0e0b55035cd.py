"""Matrix multiplication 2,2,14 - Strassen-based rank-49 decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p = 14, split into 7 blocks of size 2
    block_size = 2
    num_blocks = p // block_size  # 7
    strassen_factors = 7
    rank = num_blocks * strassen_factors  # 49

    # Strassen decomposition for 2x2 * 2x2
    # Each factor: (U_vec, V_vec, W_vec) where each is length 4
    # U: A[0,0], A[0,1], A[1,0], A[1,1]
    # V: B[0,0], B[0,1], B[1,0], B[1,1]
    # W: C[0,0], C[0,1], C[1,0], C[1,1]
    strassen = [
        # M1 = (A11+A22)(B11+B22) -> C11+1, C22+1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A21+A22)B11 -> C21+1, C22-1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A11(B12-B22) -> C12+1, C22+1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = A22(B21-B11) -> C11+1, C21+1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (A11+A12)B22 -> C11-1, C12+1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A21-A11)(B11+B12) -> C22+1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A12-A22)(B21+B22) -> C11+1
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for b in range(num_blocks):
        for s in range(strassen_factors):
            u_vec, v_vec, w_vec = strassen[s]
            # U: A has n*m = 4 entries, same for all blocks
            for idx in range(4):
                U[idx][col] = u_vec[idx]
            # V: B block b has entries at rows j*p + k for j in 0..1, k in 0..1
            # V index: j * p + k, where j is row of B (0 or 1), k is column within block
            for j in range(2):
                for k in range(2):
                    v_idx = j * p + (b * block_size + k)
                    V[v_idx][col] = v_vec[j * 2 + k]
            # W: C block b has entries at k*n + i for k in 0..1, i in 0..1
            # W index: k * n + i, where k is column of C (global), i is row
            for k in range(2):
                for i in range(2):
                    w_idx = (b * block_size + k) * n + i
                    W[w_idx][col] = w_vec[k * 2 + i]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
