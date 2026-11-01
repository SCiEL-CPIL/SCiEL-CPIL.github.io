import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # We are constructing a rank 32 decomposition.
    # This is composed of:
    # 4 blocks of M_222 (rank 7) for columns 0-1, 2-3, 4-5, 6-7
    # 1 block of M_221 (rank 4) for column 8
    # Total rank = 4*7 + 4 = 32.

    rank = 32

    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M2
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        # M6
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Cols 0-1
    add_strassen(0, 1)
    # Cols 2-3
    add_strassen(2, 3)
    # Cols 4-5
    add_strassen(4, 5)
    # Cols 6-7
    add_strassen(6, 7)
    # Col 8
    add_trivial(8)

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try to find a rank-31 decomposition via local search
    # The matrix multiplication tensor T[i,j,k] = delta_{i,k} for the (n,m,p) = (2,2,9) case
    # has entries T[i,j,k] = 1 if i==k (output row) and j is the column index in B.
    # Actually, the tensor for C = A*B where A is 2x2, B is 2x9:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) where we need A[i,j]*B[j,k] to contribute to C[i,k].
    # The tensor T has shape (n*m, m*p, p*n) = (4, 18, 18)
    # T[a, b, c] where a = i*m+j (row of A, col of A), b = j*p+k (row of B, col of B), c = k*n+i (row of C, col of C)
    # T[a,b,c] = 1 if the indices are consistent: i from a, j from a, j from b, k from b, k from c, i from c
    # i = a // m, j_a = a % m, j_b = b // p, k = b % p, k_c = c // n, i_c = c % n
    # T = 1 if j_a == j_b and k == k_c and i == i_c

    # Build the tensor as a flat list of (a,b,c) triples where T=1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    # Rank-31 search: try to find U', V', W' of shape (4,31), (18,31), (18,31)
    # such that sum_r U'[a,r]*V'[b,r]*W'[c,r] = T[a,b,c] for all (a,b,c) in tensor_entries

    # Use a simple random restart + gradient-free local search
    # Objective: sum over all tensor entries of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2

    import random
    import math

    target_rank = 31
    search_start = time.time()
    max_search_time = 60.0  # Leave 50s margin

    # Initialize with the rank-32 solution, dropping one column
    # Try dropping each of the 32 columns and see which gives the best residual
    best_residual = float('inf')
    best_init = None

    # Pre-compute the tensor as numpy arrays for speed
    try:
        import numpy as np
        use_numpy = True
        tensor_a = np.array([e[0] for e in tensor_entries])
        tensor_b = np.array([e[1] for e in tensor_entries])
        tensor_c = np.array([e[2] for e in tensor_entries])
        tensor_val = np.ones(len(tensor_entries))
    except ImportError:
        use_numpy = False

    if use_numpy:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Try dropping each column
        for drop_col in range(rank):
            if time.time() - search_start > max_search_time:
                break
            U_test = np.delete(U_np, drop_col, axis=1)
            V_test = np.delete(V_np, drop_col, axis=1)
            W_test = np.delete(W_np, drop_col, axis=1)

            # Compute residual
            # For each tensor entry (a,b,c), compute sum_r U[a,r]*V[b,r]*W[c,r]
            # Use broadcasting
            # U[tensor_a, :] has shape (N, 31)
            # V[tensor_b, :] has shape (N, 31)
            # W[tensor_c, :] has shape (N, 31)
            prod = U_np[tensor_a, :] * V_np[tensor_b, :] * W_np[tensor_c, :]
            # Drop the column
            prod = np.delete(prod, drop_col, axis=1)
            vals = prod.sum(axis=1)
            residual = np.sum((vals - tensor_val) ** 2)

            if residual < best_residual:
                best_residual = residual
                best_init = (U_test.copy(), V_test.copy(), W_test.copy())

        if best_init is not None and best_residual < 1e-10:
            # Found an exact rank-31 decomposition!
            best_U = best_init[0].tolist()
            best_V = best_init[1].tolist()
            best_W = best_init[2].tolist()
            best_rank = target_rank
            return {"U": best_U, "V": best_V, "W": best_W}

        # If no exact solution found by dropping a column, try a local search
        # Start from the best drop-column initialization
        if best_init is not None:
            U_cur = best_init[0].copy()
            V_cur = best_init[1].copy()
            W_cur = best_init[2].copy()

            # Simple coordinate descent / random perturbation
            lr = 0.01
            for iteration in range(500):
                if time.time() - search_start > max_search_time:
                    break

                # Compute gradient of residual w.r.t. U, V, W
                # residual = sum_{(a,b,c)} (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
                # d/dU[a,r] = 2 * sum_{(a',b,c): a'=a} (pred - 1) * V[b,r] * W[c,r]
                # This is expensive to compute exactly, so use a stochastic approach

                # Pick a random tensor entry
                idx = random.randint(0, len(tensor_entries) - 1)
                a, b, c = tensor_entries[idx]

                # Compute prediction for this entry
                pred = np.dot(np.dot(U_cur[a, :], V_cur[b, :]), W_cur[c, :])
                # Actually: sum_r U[a,r]*V[b,r]*W[c,r]
                pred = sum(U_cur[a, r] * V_cur[b, r] * W_cur[c, r] for r in range(target_rank))
                err = pred - 1.0

                # Update U[a, r] -= lr * err * V[b,r] * W[c,r]
                for r in range(target_rank):
                    U_cur[a, r] -= lr * err * V_cur[b, r] * W_cur[c, r]
                    V_cur[b, r] -= lr * err * U_cur[a, r] * W_cur[c, r]
                    W_cur[c, r] -= lr * err * U_cur[a, r] * V_cur[b, r]

                # Check residual every 50 iterations
                if iteration % 50 == 0:
                    total_res = 0.0
                    for (ta, tb, tc) in tensor_entries:
                        val = sum(U_cur[ta, r] * V_cur[tb, r] * W_cur[tc, r] for r in range(target_rank))
                        total_res += (val - 1.0) ** 2
                    if total_res < best_residual:
                        best_residual = total_res
                        if total_res < 1e-10:
                            best_U = U_cur.tolist()
                            best_V = V_cur.tolist()
                            best_W = W_cur.tolist()
                            best_rank = target_rank
                            return {"U": best_U, "V": best_V, "W": best_W}

    # If no rank-31 found, return the rank-32 solution
    return {"U": best_U, "V": best_V, "W": best_W}
