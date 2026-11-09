<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Schoolbook decomposition (rank = n*m*p = 36)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Try to find a lower rank decomposition using a structured approach
    # For 2x3x6, we can try to exploit the structure by splitting the 6 columns into 3 groups of 2
    # and using a rank-7 decomposition for each 2x2x2 block (Strassen), giving rank 3*7 = 21
    # But wait, the 2x3x6 matrix multiplication is:
    # C[2,6] = A[2,3] * B[3,6]
    # We can split B into 3 blocks of 3x2: B = [B0, B1, B2]
    # Then C = [A*B0, A*B1, A*B2]
    # Each A*Bb is 2x2, which can be done with Strassen rank 7
    # So total rank = 3 * 7 = 21

    # Let's implement Strassen for 2x2x2 and tile it 3 times
    # Strassen rank-7 decomposition for 2x2x2:
    # U vectors (coefficients for A_{00}, A_{01}, A_{10}, A_{11}):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [1, 0, -1, 0]
    # M7: [0, 1, 0, -1]
    #
    # V vectors (coefficients for B_{00}, B_{01}, B_{10}, B_{11}):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 1, 0, -1]
    # M4: [0, 0, 1, -1]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    #
    # W vectors (coefficients for C_{00}, C_{01}, C_{10}, C_{11}):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 0]
    # M3: [0, 1, 0, 0]
    # M4: [1, 0, 1, 0]
    # M5: [1, 0, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [0, 1, 0, 0]
    #
    # But W index is k*n + i, so for n=2:
    # C_{00} -> k=0, i=0 -> 0
    # C_{10} -> k=0, i=1 -> 1
    # C_{01} -> k=1, i=0 -> 2
    # C_{11} -> k=1, i=1 -> 3
    # So W order is [C_{00}, C_{10}, C_{01}, C_{11}] = [0, 1, 2, 3]

    # Strassen U, V, W for 2x2x2
    U_strassen = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, -1, 0],
        [0, 1, 0, -1]
    ], dtype=float)  # shape (7, 4)

    V_strassen = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)  # shape (7, 4)

    # W in order [C_{00}, C_{10}, C_{01}, C_{11}]
    W_strassen = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 0]
    ], dtype=float)  # shape (4, 7)
    # Wait, let me re-check. The W vectors above are for [C_{00}, C_{01}, C_{10}, C_{11}]
    # M1: [1, 0, 0, 1] -> C_{00}=1, C_{01}=0, C_{10}=0, C_{11}=1
    # M2: [0, 0, 1, 0] -> C_{00}=0, C_{01}=0, C_{10}=1, C_{11}=0
    # M3: [0, 1, 0, 0] -> C_{00}=0, C_{01}=1, C_{10}=0, C_{11}=0
    # M4: [1, 0, 1, 0] -> C_{00}=1, C_{01}=0, C_{10}=1, C_{11}=0
    # M5: [1, 0, 0, 0] -> C_{00}=1, C_{01}=0, C_{10}=0, C_{11}=0
    # M6: [0, 0, 0, 1] -> C_{00}=0, C_{01}=0, C_{10}=0, C_{11}=1
    # M7: [0, 1, 0, 0] -> C_{00}=0, C_{01}=1, C_{10}=0, C_{11}=0
    #
    # So in the order [C_{00}, C_{01}, C_{10}, C_{11}], W is:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 0]
    # M3: [0, 1, 0, 0]
    # M4: [1, 0, 1, 0]
    # M5: [1, 0, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [0, 1, 0, 0]
    #
    # But we need order [C_{00}, C_{10}, C_{01}, C_{11}] = [0, 1, 2, 3]
    # So we need to permute: index 0 stays, index 1 and 2 swap
    # Original: [C_{00}, C_{01}, C_{10}, C_{11}]
    # New: [C_{00}, C_{10}, C_{01}, C_{11}]
    # So new_W[:, c] = [old_W[0, c], old_W[2, c], old_W[1, c], old_W[3, c]]

    W_strassen_std = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ], dtype=float).T  # shape (4, 7), rows are C_{00}, C_{01}, C_{10}, C_{11}

    # Permute to [C_{00}, C_{10}, C_{01}, C_{11}]
    W_strassen = W_strassen_std[[0, 2, 1, 3], :]  # shape (4, 7)

    # Now build the rank-21 decomposition for 2x3x6
    # A is 2x3, B is 3x6, C is 2x6
    # Split B into 3 blocks of 3x2: B0 (cols 0-1), B1 (cols 2-3), B2 (cols 4-5)
    # For each block b, we compute A * Bb using Strassen
    # A is 2x3, but Strassen requires 2x2. We need to handle the extra column.
    # Actually, A is 2x3, not 2x2. So we can't directly use Strassen.
    # Let's reconsider.

    # Alternative: Use the fact that 2x3x6 = 2x3x2 * 3 (in terms of the p dimension)
    # But the rank might not simply multiply.

    # Let's try a different approach: use a known lower bound or try to find a rank-18 decomposition
    # For now, let's just return the schoolbook decomposition if we can't find a better one quickly.

    # Check time
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to find a rank-18 decomposition by solving a system of equations
    # This is complex, so let's just return the schoolbook for now
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
=======
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _compute_error(U, V, W, n, m, p):
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0) ** 2
    return err

def _try_round(U, V, W, n, m, p):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    err = _compute_error(U_r, V_r, W_r, n, m, p)
    if err < 1e-8:
        return U_r, V_r, W_r, True
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school

    for target_rank in [24, 21, 20, 18, 15, 12]:
        if time.time() > deadline - 30:
            break

        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break

            np.random.seed(42 + attempt * 7919)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)

            prev_err = np.inf
            for it in range(50):
                if time.time() > deadline - 5:
                    break

                for i in range(n):
                    for j in range(m):
                        A_mat = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        b_vec = np.ones(p)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[i * m + j, :] = sol

                for j in range(m):
                    for k in range(p):
                        A_mat = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        b_vec = np.ones(n)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[j * p + k, :] = sol

                for k in range(p):
                    for i in range(n):
                        A_mat = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        b_vec = np.ones(m)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[k * n + i, :] = sol

                if it % 5 == 0 or it == 49:
                    err = _compute_error(U, V, W, n, m, p)
                    if err < 1e-10:
                        found = True
                        break
                    if abs(prev_err - err) < 1e-14:
                        break
                    prev_err = err

            if found:
                U_r, V_r, W_r, ok = _try_round(U, V, W, n, m, p)
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break

        if found and best_rank == target_rank:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
>>>>>>> REPLACE
