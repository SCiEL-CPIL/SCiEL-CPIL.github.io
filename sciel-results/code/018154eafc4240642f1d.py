"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_a = n * m   # 6
    dim_b = m * p   # 18
    dim_c = p * n   # 12

    # Build target tensor T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W):
        """Check exact decomposition via einsum."""
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(result, T)

    def strassen_block_decomposition():
        """
        Rank-33 decomposition: 3 blocks, each handling 2 columns of B.
        Each block = Strassen(2x2x2) for A[:,0:2] x B[0:2, 2b:2b+2]
                  + Standard(2x1x2) for A[:,2] x B[2, 2b:2b+2]
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)

        # Strassen 2x2x2 factors (7 terms)
        S_U = np.array([
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ], dtype=np.int64)
        S_V = np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ], dtype=np.int64)
        S_W = np.array([
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ], dtype=np.int64)

        # Standard 2x1x2 factors (4 terms)
        Std_U = np.array([
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ], dtype=np.int64)
        Std_V = np.array([
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ], dtype=np.int64)
        Std_W = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ], dtype=np.int64)

        col = 0
        for b in range(3):
            # Strassen part: A cols 0,1 x B rows 0,1 cols 2b,2b+1
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r, i, j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r, j, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r, kp, i]
                col += 1

            # Standard part: A col 2 x B row 2 cols 2b,2b+1
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r, i, 0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r, 0, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r, kp, i]
                col += 1

        return U, V, W

    # Start with rank-33 baseline
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33

    # Verify baseline
    if not verify(best_U, best_V, best_W):
        # Fallback: schoolbook rank 36
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

    # Compute lower bounds from matricizations
    T1 = T.reshape(dim_a, dim_b * dim_c)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = T.reshape(dim_b, dim_a * dim_c)
    rank2 = np.linalg.matrix_rank(T2)
    T3 = T.reshape(dim_c, dim_a * dim_b)
    rank3 = np.linalg.matrix_rank(T3)
    lower_bound = max(rank1, rank2, rank3)

    # Search for lower ranks
    found_improvement = False
    search_ranks = list(range(max(lower_bound, 30), best_rank))
    search_ranks.sort()

    for r in search_ranks:
        if time.time() > deadline - 20:
            break

        max_attempts = 8000 if r <= 32 else 3000
        attempts = 0
        while not found_improvement and attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1

            # Strategy: structured sparse random with {-1, 0, 1} entries
            # Use varied sparsity patterns
            U = np.zeros((dim_a, r), dtype=np.int64)
            V = np.zeros((dim_b, r), dtype=np.int64)
            W = np.zeros((dim_c, r), dtype=np.int64)
            
            for c in range(r):
                # Vary sparsity based on attempt number for diversity
                nz_u = np.random.randint(1, 5)
                nz_v = np.random.randint(1, 6)
                nz_w = np.random.randint(1, 5)
                idx_u = np.random.choice(dim_a, size=min(nz_u, dim_a), replace=False)
                idx_v = np.random.choice(dim_b, size=min(nz_v, dim_b), replace=False)
                idx_w = np.random.choice(dim_c, size=min(nz_w, dim_c), replace=False)
                U[idx_u, c] = np.random.choice([-1, 1], size=len(idx_u))
                V[idx_v, c] = np.random.choice([-1, 1], size=len(idx_v))
                W[idx_w, c] = np.random.choice([-1, 1], size=len(idx_w))

            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found_improvement = True
                break

    # Also try a more targeted search: perturb the rank-33 solution
    # by zeroing out one column and trying to fix with a new column
    if not found_improvement and time.time() < deadline - 25:
        U33, V33, W33 = strassen_block_decomposition()
        # Try removing each column and see if we can compensate
        for remove_col in range(33):
            if time.time() > deadline - 25:
                break
            # Remove column and check if remaining 32 columns suffice
            U_trial = np.delete(U33, remove_col, axis=1)
            V_trial = np.delete(V33, remove_col, axis=1)
            W_trial = np.delete(W33, remove_col, axis=1)
            if verify(U_trial, V_trial, W_trial):
                best_U, best_V, best_W = U_trial, V_trial, W_trial
                best_rank = 32
                found_improvement = True
                break
            # Try replacing removed column with a random one
            for _ in range(200):
                if time.time() > deadline - 25:
                    break
                U_new = U_trial.copy()
                V_new = V_trial.copy()
                W_new = W_trial.copy()
                nz_u = np.random.randint(1, 5)
                nz_v = np.random.randint(1, 6)
                nz_w = np.random.randint(1, 5)
                idx_u = np.random.choice(dim_a, size=min(nz_u, dim_a), replace=False)
                idx_v = np.random.choice(dim_b, size=min(nz_v, dim_b), replace=False)
                idx_w = np.random.choice(dim_c, size=min(nz_w, dim_c), replace=False)
                U_new[idx_u, 32] = np.random.choice([-1, 1], size=len(idx_u))
                V_new[idx_v, 32] = np.random.choice([-1, 1], size=len(idx_v))
                W_new[idx_w, 32] = np.random.choice([-1, 1], size=len(idx_w))
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = 32
                    found_improvement = True
                    break
            if found_improvement:
                break

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
