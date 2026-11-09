"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # 2x2x2 Strassen rank-7 decomposition
    # U: 4x7, rows (i,j) for i,j in {0,1}
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ], dtype=float)
    # V: 4x7, rows (j,k) for j,k in {0,1}
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ], dtype=float)
    # W: 4x7, rows (k,i) for k,i in {0,1}
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ], dtype=float)

    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2

    total_rank = n_blocks * m_blocks * p_blocks * 7  # 84

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    # U: rows (i,j) for i in [bi*2, bi*2+2), j in [bj*2, bj*2+2)
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j, col] = S_U[di * 2 + dj, c]

                    # V: rows (j,k) for j in [bj*2, bj*2+2), k in [bk*2, bk*2+2)
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k, col] = S_V[dj * 2 + dk, c]

                    # W: rows (k,i) for k in [bk*2, bk*2+2), i in [bi*2, bi*2+2)
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i, col] = S_W[dk * 2 + di, c]

                col_offset += 7

    def verify_decomp(U_np, V_np, W_np):
        """Check if decomposition is valid: T[i,j,k] should be 1 for all i,j,k."""
        rank = U_np.shape[1]
        # Efficiently compute tensor using einsum
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_decomp = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        return np.allclose(T_decomp, 1.0, atol=1e-9)

    def to_int_if_possible(arr):
        """Convert numpy array to list of ints if all values are integers."""
        if np.allclose(arr, np.round(arr), atol=1e-9):
            return [[int(round(x)) for x in row] for row in arr]
        return arr.tolist()

    # Verify current decomposition
    if not verify_decomp(U, V, W):
        return {"U": to_int_if_possible(U), "V": to_int_if_possible(V), "W": to_int_if_possible(W)}

    # Try to remove columns one by one to reduce rank
    current_rank = total_rank
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    removed_cols = set()

    # Greedy column removal
    for _ in range(current_rank):
        if time.time() > deadline - 20:
            break
        best_col_to_remove = -1
        best_rank = current_rank

        for r in range(current_rank):
            if r in removed_cols:
                continue
            # Try removing column r
            keep_mask = [c for c in range(current_rank) if c != r]
            U_test = U_cur[:, keep_mask]
            V_test = V_cur[:, keep_mask]
            W_test = W_cur[:, keep_mask]

            if verify_decomp(U_test, V_test, W_test):
                best_col_to_remove = r
                best_rank = current_rank - 1
                break

        if best_col_to_remove == -1:
            break

        # Remove the column
        keep_mask = [c for c in range(current_rank) if c != best_col_to_remove]
        U_cur = U_cur[:, keep_mask]
        V_cur = V_cur[:, keep_mask]
        W_cur = W_cur[:, keep_mask]
        removed_cols.add(best_col_to_remove)
        current_rank = best_rank

    # Convert to lists with integer values if possible
    U_out = to_int_if_possible(U_cur)
    V_out = to_int_if_possible(V_cur)
    W_out = to_int_if_possible(W_cur)

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
