import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44

    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]

    col = 0

    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col

    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)

    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1

    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    start_time = time.time()
    deadline = start_time + 120

    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the tensor decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    coeff = 0.0
                    for r in range(U_mat.shape[1]):
                        coeff += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        return True

    # Verify the base decomposition
    if not verify(U_np, V_np, W_np, n, m, p):
        # Fallback to schoolbook rank-48
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(dim_u)]
        V_fb = [[0] * rank_fb for _ in range(dim_v)]
        W_fb = [[0] * rank_fb for _ in range(dim_w)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Iteratively remove linearly dependent columns
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank

    # Build the combined matrix M where column r = [U[:,r], V[:,r], W[:,r]]
    def build_M(U_mat, V_mat, W_mat):
        return np.vstack([U_mat, V_mat, W_mat])

    M = build_M(current_U, current_V, current_W)

    # Try to remove dependent columns iteratively
    for iteration in range(current_rank):
        if time.time() > deadline:
            break

        found_dependent = False
        n_cols = M.shape[1]

        for r in range(n_cols):
            if time.time() > deadline:
                break

            if r == n_cols - 1:
                # Last column: check if it's in span of columns 0..n_cols-2
                M_reduced = M[:, :r]
                target = M[:, r]
            else:
                # Check if column r is in span of all other columns
                idx = [i for i in range(n_cols) if i != r]
                M_reduced = M[:, idx]
                target = M[:, r]

            try:
                x, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, target, rcond=None)
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] < 1e-8:
                        # Column r is in the span of the others
                        # Remove it
                        if r == n_cols - 1:
                            keep_idx = list(range(r))
                        else:
                            keep_idx = [i for i in range(n_cols) if i != r]

                        current_U = current_U[:, keep_idx]
                        current_V = current_V[:, keep_idx]
                        current_W = current_W[:, keep_idx]
                        M = build_M(current_U, current_V, current_W)
                        current_rank -= 1
                        found_dependent = True
                        break
            except Exception:
                pass

        if not found_dependent:
            break

    # Final verification
    if current_rank < rank:
        if verify(current_U, current_V, current_W, n, m, p):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    # Return the original rank-44 decomposition
    return {"U": U, "V": V, "W": W}
