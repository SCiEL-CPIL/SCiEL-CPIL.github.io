import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def verify(U, V, W, r):
        """Verify the decomposition is exact."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                                expected = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                if abs(s - expected) > 1e-9:
                                    return False
        return True

    def try_rank(r, max_trials=100):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False

        # Pre-compute the block structure
        # For each (kp, ip), we have a linear system A_block * x = b_block
        # where x = W[kp, ip, :] (length r)
        # A_block has rows indexed by (j, k, i, jp) with m*p*n*m = 336 rows
        # b_block[row] = 1 if j==jp and k==kp and i==ip else 0

        # Pre-compute the index mapping for each block
        # Row index in A_block: (j, k, i, jp) -> flat index
        # j: 0..m-1, k: 0..p-1, i: 0..n-1, jp: 0..m-1
        row_order = []
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    for jp in range(m):
                        row_order.append((j, k, i, jp))

        for trial in range(max_trials):
            if time.time() > deadline:
                return False

            # Random integer U and V
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))

            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)

            W_flat = np.zeros(p * n * r)
            valid = True

            for kp in range(p):
                for ip in range(n):
                    # Build A_block: (m*p*n*m, r)
                    # A_block[row, c] = U[i, j, c] * V[jp, k, c]
                    A_rows = []
                    b_vals = []
                    for (j, k, i, jp) in row_order:
                        coeffs = U_reshaped[i, j, :] * V_reshaped[jp, k, :]
                        A_rows.append(coeffs)
                        rhs = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                        b_vals.append(rhs)

                    A_block = np.array(A_rows)
                    b_block = np.array(b_vals)

                    # Solve least squares
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                        residual = A_block @ x - b_block
                        if np.max(np.abs(residual)) < 1e-6:
                            # Check integrality
                            if np.all(np.abs(x - np.round(x)) < 1e-4):
                                W_block = np.round(x).astype(int)
                                W_flat[kp * n * r + ip * r : (kp * n + ip + 1) * r] = W_block
                            else:
                                # Allow half-integers
                                x_half = 2 * x
                                if np.all(np.abs(x_half - np.round(x_half)) < 1e-4):
                                    W_block = np.round(x_half).astype(int) / 2.0
                                    W_flat[kp * n * r + ip * r : (kp * n + ip + 1) * r] = W_block
                                else:
                                    valid = False
                                    break
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break

                if not valid:
                    break

            if valid:
                W_mat = W_flat.reshape(p * n, r)
                U_int = U.astype(int)
                V_int = V.astype(int)

                # Verify the full decomposition
                if verify(U_int, V_int, W_mat, r):
                    best_rank = r
                    best_U = U_int
                    best_V = V_int
                    best_W = W_mat
                    return True

        return False

    # Try ranks from 20 upward
    for r in range(20, best_rank + 1):
        if time.time() > deadline:
            break
        if try_rank(r, max_trials=50):
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
