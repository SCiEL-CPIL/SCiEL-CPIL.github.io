import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W):
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def strassen_block_decomposition():
        """Rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2)."""
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)

        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]

        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]

        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                col += 1

            # Standard part: handles column j=2 of A/B
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                col += 1

        return U, V, W

    # Start with rank-33 baseline
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33

    if not verify(best_U, best_V, best_W):
        # Fallback: schoolbook
        best_rank = n * m * p
        best_U = np.zeros((dim_a, best_rank), dtype=np.int64)
        best_V = np.zeros((dim_b, best_rank), dtype=np.int64)
        best_W = np.zeros((dim_c, best_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    best_U[a, col] = 1
                    best_V[b, col] = 1
                    best_W[c, col] = 1
                    col += 1

    # Lower bound from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for rank-32 by trying to remove one column from rank-33
    # and solving for it as a linear combination of the others
    if time.time() < deadline - 20:
        found = False
        for remove_idx in range(33):
            if found or time.time() > deadline - 20:
                break
            # Try to express column remove_idx as linear combo of others
            # T = sum_c u_c v_c^T w_c^T
            # If we remove column r, we need: u_r v_r^T w_r^T = sum_{c != r} alpha_c u_c v_c^T w_c^T
            # This is a tensor equation. Instead, try small perturbations.

            # Try: set one coefficient in the removed column to vary, adjust others
            # Simpler: try replacing column r with a modified version
            for sign in [1, -1]:
                if found or time.time() > deadline - 20:
                    break
                U_test = best_U.copy()
                V_test = best_V.copy()
                W_test = best_W.copy()
                # Remove column r and see if we can find a replacement
                # Try: new column = old column + small perturbation in one entry
                for perturb_dim in range(3):
                    if found or time.time() > deadline - 20:
                        break
                    mat = [U_test, V_test, W_test][perturb_dim]
                    dim_size = [dim_a, dim_b, dim_c][perturb_dim]
                    for entry in range(dim_size):
                        if found or time.time() > deadline - 20:
                            break
                        U_t2 = best_U.copy()
                        V_t2 = best_V.copy()
                        W_t2 = best_W.copy()
                        # Modify entry in the removed column
                        if perturb_dim == 0:
                            U_t2[entry, remove_idx] += sign
                        elif perturb_dim == 1:
                            V_t2[entry, remove_idx] += sign
                        else:
                            W_t2[entry, remove_idx] += sign
                        # Try to compensate by modifying another column
                        for other_idx in range(33):
                            if other_idx == remove_idx or time.time() > deadline - 20:
                                continue
                            for other_dim in range(3):
                                if found or time.time() > deadline - 20:
                                    break
                                mat2 = [U_t2, V_t2, W_t2][other_dim]
                                dim_size2 = [dim_a, dim_b, dim_c][other_dim]
                                for entry2 in range(dim_size2):
                                    if found or time.time() > deadline - 20:
                                        break
                                    U_t3 = U_t2.copy()
                                    V_t3 = V_t2.copy()
                                    W_t3 = W_t2.copy()
                                    if other_dim == 0:
                                        U_t3[entry2, other_idx] += sign
                                    elif other_dim == 1:
                                        V_t3[entry2, other_idx] += sign
                                    else:
                                        W_t3[entry2, other_idx] += sign
                                    # Remove column remove_idx
                                    mask = [i for i in range(33) if i != remove_idx]
                                    U_final = U_t3[:, mask]
                                    V_final = V_t3[:, mask]
                                    W_final = W_t3[:, mask]
                                    if verify(U_final, V_final, W_final):
                                        best_U, best_V, best_W = U_final, V_final, W_final
                                        best_rank = 32
                                        found = True
                                        break

    # Random search for rank 32
    if best_rank > 32 and time.time() < deadline - 15:
        np.random.seed(42)
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            r = 32
            # Sparse random search
            U = np.zeros((dim_a, r), dtype=np.int64)
            V = np.zeros((dim_b, r), dtype=np.int64)
            W = np.zeros((dim_c, r), dtype=np.int64)
            for c in range(r):
                for _ in range(np.random.randint(1, 4)):
                    U[np.random.randint(dim_a), c] = np.random.choice([-1, 1])
                for _ in range(np.random.randint(1, 4)):
                    V[np.random.randint(dim_b), c] = np.random.choice([-1, 1])
                for _ in range(np.random.randint(1, 4)):
                    W[np.random.randint(dim_c), c] = np.random.choice([-1, 1])
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = 32
                break

    # Convert to list of lists for the required interface
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
