def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
    add_strassen_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    add_strassen_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
    add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])

    # Attempt rank-27: try to find a linear combination that eliminates one term
    # by adjusting coefficients in a non-integer way, then rounding.
    # This is a heuristic; may not succeed.
    if time.time() - start_time < DEADLINE - 20:
        try:
            import numpy as np
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)

            # Try to find a rank-27 solution by solving a least-squares problem
            # with one column forced to zero.
            # For each column c to eliminate, try to find adjustments to other columns.
            for c in range(rank):
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Create design: we want sum_{r != c} u_r * v_r * w_r = T - u_c * v_c * w_c
                # This is nonlinear. Skip for now.
                pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
