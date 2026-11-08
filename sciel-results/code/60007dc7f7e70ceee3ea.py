import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 110

    # Schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try lower ranks
    target_ranks = [55, 60, 65, 70, 75, 80, 85, 90, 95]

    for R in target_ranks:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue

        for attempt in range(30):
            if time.time() > deadline:
                break

            # Randomly initialize U, V with small integer/half-integer values
            U = np.random.randint(-2, 3, (n * m, R)).astype(float)
            V = np.random.randint(-2, 3, (m * p, R)).astype(float)

            # Construct linear system for W
            # Equations: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
            # Variables: W[k*n+i, r]
            # Number of equations: n*m*p = 96
            # Number of variables: p*n*R

            A_mat = np.zeros((n * m * p, p * n * R))
            b_vec = np.ones(n * m * p)

            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            var_idx = (k * n + i) * R + r
                            A_mat[idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                        idx += 1

            # Solve A_mat @ W_flat = b_vec
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except Exception:
                continue

            # Check if solution is exact
            if np.allclose(A_mat @ W_flat, b_vec, atol=1e-9):
                W = W_flat.reshape((p * n, R))
                # Snap to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2

                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_val = 0.0
                            for r in range(R):
                                s_val += U_int[i * m + j, r] * V_int[j * p + k, r] * W_int[k * n + i, r]
                            if abs(s_val - 1) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break

                if valid:
                    best_rank = R
                    best_U, best_V, best_W = U_int, V_int, W_int
                    break  # Found a valid decomposition for this R

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
