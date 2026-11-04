"""Combined approach: structural embedding + rank reduction via linear dependence search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    # Start with the known rank-55 decomposition (Strassen + schoolbook)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},       # M1: a00, a11
        {2: 1, 3: 1},       # M2: a10, a11
        {0: 1},             # M3: a00
        {3: 1},             # M4: a11
        {0: 1, 1: 1},       # M5: a00, a01
        {2: 1, 0: -1},      # M6: a10, -a00
        {1: 1, 3: -1}       # M7: a01, -a11
    ]
    R = [
        {0: 1, 3: 1},       # M1: b00, b11
        {0: 1},             # M2: b00
        {1: 1, 3: -1},      # M3: b01, -b11
        {2: 1, 0: -1},      # M4: b10, -b00
        {3: 1},             # M5: b11
        {0: 1, 1: 1},       # M6: b00, b01
        {2: 1, 3: 1}        # M7: b10, b11
    ]
    Wc = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]

    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1

    # Schoolbook for the 2x1 * 1x10 part
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # Verification: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # Let's build the tensor and verify
    def build_tensor(U_arr, V_arr, W_arr):
        """Build the 3D tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[i, j, k] = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
        return T

    def is_valid(U_arr, V_arr, W_arr, tol=1e-9):
        T = build_tensor(U_arr, V_arr, W_arr)
        return np.allclose(T, 1.0, atol=tol)

    # Verify initial decomposition
    if not is_valid(U_np, V_np, W_np):
        # Return the initial decomposition as fallback
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by finding linear dependencies
    # For each column r, check if removing it still gives a valid tensor
    # This is O(rank * n*m*p) per check, which is manageable for rank=55

    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()

    # Greedy removal: try to remove each column one at a time
    # If removing column r keeps the tensor valid, remove it
    removed = set()
    changed = True
    while changed and time.time() - start_time < deadline - 10:
        changed = False
        for r in range(current_rank):
            if r in removed:
                continue
            # Build tensor without column r
            U_test = np.delete(current_U, r, axis=1)
            V_test = np.delete(current_V, r, axis=1)
            W_test = np.delete(current_W, r, axis=1)
            if is_valid(U_test, V_test, W_test):
                # Remove column r
                current_U = U_test
                current_V = V_test
                current_W = W_test
                removed.add(r)
                current_rank -= 1
                changed = True
                break  # Restart the loop since indices changed

    # Also try: can we find a linear combination of columns that sums to zero?
    # This would allow us to reduce rank further
    # Check if any column is a linear combination of others
    # For efficiency, check small subsets

    if time.time() - start_time < deadline - 20:
        # Try to find if any 2 columns sum to a third (or similar simple relations)
        # This is a heuristic for finding more dependencies
        rank_now = current_rank
        for r1 in range(rank_now):
            if time.time() - start_time > deadline - 15:
                break
            for r2 in range(r1 + 1, rank_now):
                # Check if col r1 + col r2 = col r3 for some r3
                # Actually, let's just check if the sum of two columns is zero
                # (i.e., they are negatives of each other)
                sum_U = current_U[:, r1] + current_U[:, r2]
                sum_V = current_V[:, r1] + current_V[:, r2]
                sum_W = current_W[:, r1] + current_W[:, r2]
                if np.allclose(sum_U, 0) and np.allclose(sum_V, 0) and np.allclose(sum_W, 0):
                    # These two columns cancel out, remove both
                    current_U = np.delete(current_U, [r1, r2], axis=1)
                    current_V = np.delete(current_V, [r1, r2], axis=1)
                    current_W = np.delete(current_W, [r1, r2], axis=1)
                    current_rank -= 2
                    break
            if time.time() - start_time > deadline - 15:
                break

    # Convert back to lists
    U_final = current_U.tolist()
    V_final = current_V.tolist()
    W_final = current_W.tolist()

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
