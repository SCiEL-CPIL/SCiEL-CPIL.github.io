import time
import random

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Build the rank-32 base solution
    # 4 blocks of M_222 (rank 7) for columns 0-1, 2-3, 4-5, 6-7
    # 1 block of M_221 (rank 4) for column 8
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

    # Build the tensor as a flat list of (a,b,c) triples where T=1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    # Pre-compute the tensor as numpy arrays for speed
    import numpy as np

    tensor_a = np.array([e[0] for e in tensor_entries])
    tensor_b = np.array([e[1] for e in tensor_entries])
    tensor_c = np.array([e[2] for e in tensor_entries])
    tensor_val = np.ones(len(tensor_entries))
    N = len(tensor_entries)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def compute_residual(U_arr, V_arr, W_arr):
        """Compute sum of squared residuals for all tensor entries."""
        prod = U_arr[tensor_a, :] * V_arr[tensor_b, :] * W_arr[tensor_c, :]
        vals = prod.sum(axis=1)
        return np.sum((vals - tensor_val) ** 2)

    def compute_residual_max(U_arr, V_arr, W_arr):
        """Compute max absolute residual for all tensor entries."""
        prod = U_arr[tensor_a, :] * V_arr[tensor_b, :] * W_arr[tensor_c, :]
        vals = prod.sum(axis=1)
        return np.max(np.abs(vals - tensor_val))

    # First, verify the rank-32 solution is exact
    res32 = compute_residual(U_np, V_np, W_np)
    if res32 < 1e-10:
        # Return the valid rank-32 solution immediately if search fails
        base_solution = {"U": U, "V": V, "W": W}
    else:
        # Should not happen, but just in case
        base_solution = {"U": U, "V": V, "W": W}

    # Try to find a rank-31 solution by dropping one column and polishing
    search_start = time.time()
    max_search_time = 90.0  # Leave 20s margin for final processing

    # Try dropping each column and see if we can find an exact rank-31 solution
    # by adjusting the remaining 31 columns slightly

    # First, try simple dropping - check if any single column can be dropped
    # and the remaining 31 still give an exact decomposition
    best_rank31_solution = None

    for drop_col in range(rank):
        if time.time() - search_start > max_search_time:
            break

        U_test = np.delete(U_np, drop_col, axis=1)
        V_test = np.delete(V_np, drop_col, axis=1)
        W_test = np.delete(W_np, drop_col, axis=1)

        res = compute_residual(U_test, V_test, W_test)
        if res < 1e-10:
            # Found an exact rank-31 solution by simple dropping
            best_rank31_solution = {
                "U": U_test.tolist(),
                "V": V_test.tolist(),
                "W": W_test.tolist()
            }
            break

    # If simple dropping didn't work, try local search with small perturbations
    if best_rank31_solution is None:
        # Try gradient-based optimization to see if we can find a rank-31 solution
        # Start from the best rank-32 solution and try to reduce rank

        for drop_col in range(rank):
            if time.time() - search_start > max_search_time:
                break

            U_test = np.delete(U_np, drop_col, axis=1)
            V_test = np.delete(V_np, drop_col, axis=1)
            W_test = np.delete(W_np, drop_col, axis=1)

            # Compute initial residual
            res0 = compute_residual(U_test, V_test, W_test)
            if res0 < 1e-10:
                best_rank31_solution = {
                    "U": U_test.tolist(),
                    "V": V_test.tolist(),
                    "W": W_test.tolist()
                }
                break

            # Try gradient descent with small steps
            # Use a simplified gradient approach
            lr = 0.001
            prev_res = res0
            improved = False

            for iteration in range(500):
                if time.time() - search_start > max_search_time:
                    break

                # Compute predictions
                prod = U_test[tensor_a, :] * V_test[tensor_b, :] * W_test[tensor_c, :]
                pred = prod.sum(axis=1)
                err = pred - tensor_val  # shape (N,)

                # Compute gradients (simplified)
                U_grad = np.zeros_like(U_test)
                V_grad = np.zeros_like(V_test)
                W_grad = np.zeros_like(W_test)

                R = U_test.shape[1]
                V_idx = V_test[tensor_b, :] # (N, R)
                W_idx = W_test[tensor_c, :] # (N, R)
                U_idx = U_test[tensor_a, :] # (N, R)

                for r in range(R):
                    contrib_U_r = err * V_idx[:, r] * W_idx[:, r]
                    np.add.at(U_grad[:, r], tensor_a, contrib_U_r)

                    contrib_V_r = err * U_idx[:, r] * W_idx[:, r]
                    np.add.at(V_grad[:, r], tensor_b, contrib_V_r)

                    contrib_W_r = err * U_idx[:, r] * V_idx[:, r]
                    np.add.at(W_grad[:, r], tensor_c, contrib_W_r)

                # Update
                U_test = U_test - lr * U_grad
                V_test = V_test - lr * V_grad
                W_test = W_test - lr * W_grad

                if iteration % 10 == 0:
                    res = compute_residual(U_test, V_test, W_test)
                    if res < 1e-10:
                        improved = True
                        break
                    if res < prev_res:
                        lr = min(lr * 1.05, 0.01)
                    else:
                        lr = lr * 0.9
                    prev_res = res

            if improved:
                best_rank31_solution = {
                    "U": U_test.tolist(),
                    "V": V_test.tolist(),
                    "W": W_test.tolist()
                }
                break

    # Return the best solution found
    if best_rank31_solution is not None:
        return best_rank31_solution

    # If no rank-31 found, return the rank-32 solution
    return base_solution
