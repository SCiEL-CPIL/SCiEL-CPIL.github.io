import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Pre-compute index arrays for vectorized evaluation
    # Each tensor index (i,j,k) maps to U[i*m+j], V[j*p+k], W[k*n+i]
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_indices = len(indices)

    # For each index, store the row indices into U, V, W
    u_rows = np.array([i * m + j for i, j, k in indices])
    v_rows = np.array([j * p + k for i, j, k in indices])
    w_rows = np.array([k * n + i for i, j, k in indices])

    # Schoolbook fallback (rank = n*m*p = 112)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Vectorized evaluation: given U, V, W as numpy arrays of shape (rows, R),
    # compute the tensor values T[i,j,k] = sum_r U[u_rows[i,j,k], r] * V[v_rows[i,j,k], r] * W[w_rows[i,j,k], r]
    def evaluate(U_arr, V_arr, W_arr):
        # U_arr: (n*m, R), V_arr: (m*p, R), W_arr: (p*n, R)
        # For each index t in range(num_indices), compute sum over r
        # U_sel = U_arr[u_rows, :]  shape (num_indices, R)
        # V_sel = V_arr[v_rows, :]  shape (num_indices, R)
        # W_sel = W_arr[w_rows, :]  shape (num_indices, R)
        # T = sum(U_sel * V_sel * W_sel, axis=1)
        U_sel = U_arr[u_rows]
        V_sel = V_arr[v_rows]
        W_sel = W_arr[w_rows]
        return np.sum(U_sel * V_sel * W_sel, axis=1)

    def is_valid(T):
        return bool(np.all(T == 1))

    def get_error_indices(T):
        return np.where(T != 1)[0]

    # Try ranks in descending order from near schoolbook down to lower bound
    # Lower bound: max(n*m, m*p, n*p) = max(16, 28, 28) = 28
    # We'll try a range of ranks, spending more time on promising ones
    ranks_to_try = list(range(110, 27, -1))  # 110 down to 28

    best_solution = None
    best_rank = rank_school

    for R in ranks_to_try:
        if time.time() > deadline - 5:
            break

        if R >= best_rank:
            continue

        # Number of restarts: more for lower ranks (harder), fewer for higher
        if R <= 40:
            max_restarts = 15
        elif R <= 60:
            max_restarts = 8
        else:
            max_restarts = 3

        found = False

        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            # Initialize with random small integers, biased toward 0, 1, -1
            # Use numpy for fast initialization
            U_arr = np.array([[random.choice([0, 0, 1, -1]) for _ in range(R)] for _ in range(n * m)])
            V_arr = np.array([[random.choice([0, 0, 1, -1]) for _ in range(R)] for _ in range(m * p)])
            W_arr = np.array([[random.choice([0, 0, 1, -1]) for _ in range(R)] for _ in range(p * n)])

            # Local search with vectorized evaluation
            max_iters = 2000
            prev_error_count = num_indices + 1

            for it in range(max_iters):
                if time.time() > deadline - 1:
                    break

                T = evaluate(U_arr, V_arr, W_arr)
                err_idx = get_error_indices(T)
                error_count = len(err_idx)

                if error_count == 0:
                    # Valid solution found
                    best_rank = R
                    best_solution = {
                        "U": U_arr.tolist(),
                        "V": V_arr.tolist(),
                        "W": W_arr.tolist()
                    }
                    found = True
                    break

                # If error count hasn't improved in a while, restart local search
                if error_count >= prev_error_count and it > 100:
                    if error_count == prev_error_count and it > 500:
                        break
                prev_error_count = error_count

                # Pick a random error index
                t = random.choice(err_idx.tolist())
                i, j, k = indices[t]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                current_sum = int(T[t])

                # Try to fix by adjusting one variable in a random term r
                r = random.randint(0, R - 1)

                u_r = int(U_arr[u_idx, r])
                v_r = int(V_arr[v_idx, r])
                w_r = int(W_arr[w_idx, r])
                contrib = u_r * v_r * w_r

                # Target contribution: new_sum = current_sum - contrib + new_contrib = 1
                # new_contrib = 1 - current_sum + contrib
                target_contrib = 1 - current_sum + contrib

                # Try adjusting U
                if v_r != 0 and w_r != 0:
                    denom = v_r * w_r
                    if target_contrib % denom == 0:
                        new_u = target_contrib // denom
                        if new_u != u_r and -15 <= new_u <= 15:
                            U_arr[u_idx, r] = new_u
                            continue

                # Try adjusting V
                if u_r != 0 and w_r != 0:
                    denom = u_r * w_r
                    if target_contrib % denom == 0:
                        new_v = target_contrib // denom
                        if new_v != v_r and -15 <= new_v <= 15:
                            V_arr[v_idx, r] = new_v
                            continue

                # Try adjusting W
                if u_r != 0 and v_r != 0:
                    denom = u_r * v_r
                    if target_contrib % denom == 0:
                        new_w = target_contrib // denom
                        if new_w != w_r and -15 <= new_w <= 15:
                            W_arr[w_idx, r] = new_w
                            continue

                # If no single-variable fix worked, try a random perturbation
                # to escape local minima
                if random.random() < 0.1:
                    var = random.choice(['U', 'V', 'W'])
                    if var == 'U':
                        row = random.randint(0, n * m - 1)
                        col_r = random.randint(0, R - 1)
                        U_arr[row, col_r] = random.choice([0, 1, -1, 2, -2])
                    elif var == 'V':
                        row = random.randint(0, m * p - 1)
                        col_r = random.randint(0, R - 1)
                        V_arr[row, col_r] = random.choice([0, 1, -1, 2, -2])
                    else:
                        row = random.randint(0, p * n - 1)
                        col_r = random.randint(0, R - 1)
                        W_arr[row, col_r] = random.choice([0, 1, -1, 2, -2])

            # End local search
            if found:
                break

        # End restarts
        if found:
            break

    if best_solution is not None:
        return best_solution

    return {"U": U_school, "V": V_school, "W": W_school}
