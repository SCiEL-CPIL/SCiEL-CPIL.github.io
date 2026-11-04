import numpy as np
import time

def construct():
    a_rows = 2
    b_cols = 7
    mid_dim = 3
    R_A = a_rows * mid_dim  # 6
    R_B = mid_dim * b_cols  # 21
    R_C = b_cols * a_rows  # 14

    t0 = time.perf_counter()
    safety_margin = 120.0

    def build_baseline():
        """Schoolbook rank-42 decomposition."""
        total_rank = a_rows * mid_dim * b_cols
        mat_a = np.zeros((R_A, total_rank), dtype=float)
        mat_b = np.zeros((R_B, total_rank), dtype=float)
        mat_c = np.zeros((R_C, total_rank), dtype=float)
        col_idx = 0
        for i in range(a_rows):
            for j in range(mid_dim):
                for k in range(b_cols):
                    mat_a[i * mid_dim + j, col_idx] = 1.0
                    mat_b[j * b_cols + k, col_idx] = 1.0
                    mat_c[k * a_rows + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": mat_a.tolist(), "V": mat_b.tolist(), "W": mat_c.tolist()}

    def build_block_232():
        """Return (U, V, W) for 2x3x2 multiplication with rank 11."""
        # Decompose 2x3x2 as: Strassen(2x2x2) for j=0,1 + 2x1x2 for j=2
        # Strassen part: rank 7
        # 2x1x2 part: rank 4
        # Total: 11

        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, -1]
        ], dtype=float)
        V_s = np.array([
            [1, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1],
            [1, 1, -1, 0, 1, 0, 1]
        ], dtype=float)
        W_s = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)

        U_112 = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=float)
        V_112 = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ], dtype=float)
        W_112 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Assemble 2x3x2: U is 6x11, V is 9x11, W is 4x11
        U_full = np.zeros((6, 11), dtype=float)
        V_full = np.zeros((9, 11), dtype=float)
        W_full = np.zeros((4, 11), dtype=float)

        # Strassen part for j=0,1 (first two columns of A, first two rows of B)
        # U rows: (i,j) for i=0,1; j=0,1 -> rows 0,1,3,4
        U_full[0, 0:7] = U_s[0, :]  # i=0,j=0
        U_full[1, 0:7] = U_s[1, :]  # i=0,j=1
        U_full[3, 0:7] = U_s[2, :]  # i=1,j=0
        U_full[4, 0:7] = U_s[3, :]  # i=1,j=1

        # V rows: (j,k) for j=0,1; k=0,1 -> rows 0,1,2,3
        V_full[0, 0:7] = V_s[0, :]  # j=0,k=0
        V_full[1, 0:7] = V_s[1, :]  # j=0,k=1
        V_full[2, 0:7] = V_s[2, :]  # j=1,k=0
        V_full[3, 0:7] = V_s[3, :]  # j=1,k=1

        # W rows: (k,i) for k=0,1; i=0,1 -> rows 0,1,2,3
        W_full[0, 0:7] = W_s[0, :]  # k=0,i=0
        W_full[1, 0:7] = W_s[1, :]  # k=0,i=1
        W_full[2, 0:7] = W_s[2, :]  # k=1,i=0
        W_full[3, 0:7] = W_s[3, :]  # k=1,i=1

        # 2x1x2 part for j=2
        U_full[2, 7:11] = U_112[0, :]  # i=0,j=2
        U_full[5, 7:11] = U_112[1, :]  # i=1,j=2

        V_full[4, 7:11] = V_112[0, :]  # j=2,k=0
        V_full[5, 7:11] = V_112[1, :]  # j=2,k=1

        W_full[0, 7:11] = W_112[0, :]  # k=0,i=0
        W_full[1, 7:11] = W_112[1, :]  # k=0,i=1
        W_full[2, 7:11] = W_112[2, :]  # k=1,i=0
        W_full[3, 7:11] = W_112[3, :]  # k=1,i=1

        return U_full, V_full, W_full

    def build_block_231():
        """Return (U, V, W) for 2x3x1 multiplication with rank 6."""
        U = np.zeros((6, 6), dtype=float)
        V = np.zeros((3, 6), dtype=float)
        W = np.zeros((2, 6), dtype=float)
        for t in range(6):
            i = t // 3
            j = t % 3
            U[i * 3 + j, t] = 1.0
            V[j, t] = 1.0
            W[i, t] = 1.0
        return U, V, W

    def verify_exact(U, V, W):
        """Check that U @ diag(V) @ W^T reproduces the multiplication tensor."""
        tensor_computed = np.einsum('ar,br,cr->abc', U, V, W)
        tensor_expected = np.zeros((R_A, R_B, R_C), dtype=float)
        for i in range(a_rows):
            for j in range(mid_dim):
                for k in range(b_cols):
                    tensor_expected[i * mid_dim + j, j * b_cols + k, k * a_rows + i] = 1.0
        return np.allclose(tensor_computed, tensor_expected, atol=1e-10)

    # Partition 7 = 2 + 2 + 2 + 1
    # Three 2x3x2 blocks (rank 11 each) + one 2x3x1 block (rank 6) = 39
    if time.perf_counter() - t0 > safety_margin - 5:
        return build_baseline()

    total_rank = 11 + 11 + 11 + 6
    U_total = np.zeros((R_A, total_rank), dtype=float)
    V_total = np.zeros((R_B, total_rank), dtype=float)
    W_total = np.zeros((R_C, total_rank), dtype=float)

    # Block offsets: cols 0-1, 2-3, 4-5, 6 of B correspond to cols 0-1, 2-3, 4-5, 6 of C
    blocks = [
        (0, 2),   # 2x3x2, cols 0-1
        (11, 22), # 2x3x2, cols 2-3
        (22, 33), # 2x3x2, cols 4-5
    ]

    U_blk, V_blk, W_blk = build_block_232()
    for block_idx, (col_start, rank_off) in enumerate(blocks):
        k_base = block_idx * 2  # 0, 2, 4
        U_total[:, rank_off:rank_off + 11] = U_blk

        # V: for each j in 0..2, k in 0..1
        for j in range(mid_dim):
            for dk in range(2):
                local_row = j * 2 + dk
                global_row = j * b_cols + (k_base + dk)
                V_total[global_row, rank_off:rank_off + 11] = V_blk[local_row, :]

        # W: for each k in 0..1, i in 0..1
        for dk in range(2):
            for i in range(a_rows):
                local_row = dk * a_rows + i
                global_row = (k_base + dk) * a_rows + i
                W_total[global_row, rank_off:rank_off + 11] = W_blk[local_row, :]

    # Last block: 2x3x1 for col 6
    U_last, V_last, W_last = build_block_231()
    rank_off_last = 33
    U_total[:, rank_off_last:rank_off_last + 6] = U_last
    for j in range(mid_dim):
        V_total[j * b_cols + 6, rank_off_last:rank_off_last + 6] = V_last[j, :]
    for i in range(a_rows):
        W_total[6 * a_rows + i, rank_off_last:rank_off_last + 6] = W_last[i, :]

    if verify_exact(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

    return build_baseline()
