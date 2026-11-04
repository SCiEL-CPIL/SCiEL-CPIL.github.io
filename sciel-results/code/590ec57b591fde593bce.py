"""
Rank-minimal tensor decomposition for 4x5 @ 5x6 matrix multiplication.

The multiplication tensor T_{i,j,k} = 1 for all i in [0,4), j in [0,5), k in [0,6).
We seek U (20 x R), V (30 x R), W (24 x R) with integer/half-integer entries such that
sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all (i,j,k).

Strategy:
1. Fast incremental error computation: precompute per-column contributions so
   changing one entry only requires O(NMP) work, not O(R*NMP).
2. Smarter initialization using structured patterns + random perturbations.
3. Local search with incremental error updates and exact verification.
4. Time-aware budget management.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    target_rank = N * M * P  # 120
    total_terms = N * M * P  # 120

    # Precompute index arrays
    idx_i = np.repeat(np.arange(N), M * P)
    idx_j = np.tile(np.repeat(np.arange(M), P), N)
    idx_k = np.tile(np.arange(P), N * M)

    U_rows = idx_i * M + idx_j  # shape (120,)
    V_rows = idx_j * P + idx_k  # shape (120,)
    W_rows = idx_k * N + idx_i  # shape (120,)

    def schoolbook():
        U = [[0] * target_rank for _ in range(N * M)]
        V = [[0] * target_rank for _ in range(M * P)]
        W = [[0] * target_rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, R):
        """Exact verification using integer arithmetic."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        vals = np.zeros(total_terms, dtype=np.int64)
        for r in range(R):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            prod = u_col[U_rows] * v_col[V_rows] * w_col[W_rows]
            vals += prod
        return np.all(vals == 1)

    def compute_vals(U_np, V_np, W_np, R):
        """Compute the tensor values for all (i,j,k)."""
        vals = np.zeros(total_terms, dtype=np.float64)
        for r in range(R):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            prod = u_col[U_rows] * v_col[V_rows] * w_col[W_rows]
            vals += prod
        return vals

    def compute_sqr_err(vals):
        """Compute squared error from precomputed values."""
        err = vals - 1.0
        return np.sum(err * err)

    def local_search_fast(U, V, W, R, max_steps=100000):
        """
        Local search with incremental error computation.
        Changing U[row,col] from old to new changes vals by:
          (new - old) * V[V_rows, col] * W[W_rows, col]
        This is O(total_terms) per evaluation instead of O(R * total_terms).
        """
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        # Precompute column vectors for fast incremental updates
        # For each column r, precompute the "interaction" vectors
        U_cols = [U_np[:, r] for r in range(R)]
        V_cols = [V_np[:, r] for r in range(R)]
        W_cols = [W_np[:, r] for r in range(R)]

        # Precompute interaction products for each column
        # interact[r] = V_col[r][V_rows] * W_col[r][W_rows]  for U changes
        # For V changes: U_col[r][U_rows] * W_col[r][W_rows]
        # For W changes: U_col[r][U_rows] * V_col[r][V_rows]
        U_interact = np.zeros((R, total_terms), dtype=np.float64)
        V_interact = np.zeros((R, total_terms), dtype=np.float64)
        W_interact = np.zeros((R, total_terms), dtype=np.float64)
        for r in range(R):
            U_interact[r] = V_cols[r][V_rows] * W_cols[r][W_rows]
            V_interact[r] = U_cols[r][U_rows] * W_cols[r][W_rows]
            W_interact[r] = U_cols[r][U_rows] * V_cols[r][V_rows]

        cur_vals = compute_vals(U_np, V_np, W_np, R)
        cur_err = compute_sqr_err(cur_vals)

        if cur_err < 1e-12:
            if verify(U, V, W, R):
                return U, V, W, 0.0

        step = 0
        no_improve_count = 0
        candidates = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]

        while step < max_steps and time.time() < deadline - 5:
            step += 1
            improved = False

            # Choose which matrix to modify
            mat_choice = random.random()
            if mat_choice < 0.33:
                mat_np = U_np
                mat_list = U
                n_rows = N * M
                interact = U_interact
            elif mat_choice < 0.66:
                mat_np = V_np
                mat_list = V
                n_rows = M * P
                interact = V_interact
            else:
                mat_np = W_np
                mat_list = W
                n_rows = P * N
                interact = W_interact

            row = random.randrange(n_rows)
            col = random.randrange(R)
            old_val = mat_np[row, col]

            best_candidate = None
            best_new_err = cur_err

            for new_val in candidates:
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Incremental update: new_vals = cur_vals + delta * interact[col]
                new_vals = cur_vals + delta * interact[col]
                new_err = compute_sqr_err(new_vals)
                if new_err < best_new_err - 1e-15:
                    best_new_err = new_err
                    best_candidate = new_val

            if best_candidate is not None:
                delta = best_candidate - old_val
                mat_np[row, col] = best_candidate
                mat_list[row][col] = best_candidate

                # Update the interaction vectors that depend on this matrix
                if mat_np is U_np:
                    # U changed: update V_interact and W_interact for column col
                    # V_interact[col] = U_col * W_col[W_rows]
                    # W_interact[col] = U_col * V_col[V_rows]
                    V_interact[col] = U_np[:, col][U_rows] * W_cols[col][W_rows]
                    W_interact[col] = U_np[:, col][U_rows] * V_cols[col][V_rows]
                    # U_interact doesn't change (depends on V and W)
                elif mat_np is V_np:
                    U_interact[col] = V_np[:, col][V_rows] * W_cols[col][W_rows]
                    W_interact[col] = U_cols[col][U_rows] * V_np[:, col][V_rows]
                else:
                    U_interact[col] = V_cols[col][V_rows] * W_np[:, col][W_rows]
                    V_interact[col] = U_cols[col][U_rows] * W_np[:, col][W_rows]

                # Update cur_vals incrementally
                cur_vals += delta * interact[col]
                cur_err = best_new_err
                improved = True
                no_improve_count = 0

                # Check for exact solution
                if cur_err < 1e-12:
                    U_int = [[round(U_np[i, r] * 2) / 2 for r in range(R)] for i in range(N * M)]
                    V_int = [[round(V_np[i, r] * 2) / 2 for r in range(R)] for i in range(M * P)]
                    W_int = [[round(W_np[i, r] * 2) / 2 for r in range(R)] for i in range(P * N)]
                    if verify(U_int, V_int, W_int, R):
                        return U_int, V_int, W_int, 0.0
            else:
                no_improve_count += 1
                if no_improve_count > 10000:
                    break

        U_out = [[U_np[i, r] for r in range(R)] for i in range(N * M)]
        V_out = [[V_np[i, r] for r in range(R)] for i in range(M * P)]
        W_out = [[W_np[i, r] for r in range(R)] for i in range(P * N)]
        return U_out, V_out, W_out, cur_err

    def try_reduce_rank(R_target, seed=None):
        """
        Try to find a valid decomposition of rank R_target.
        """
        if seed is not None:
            random.seed(seed)

        U = [[0] * R_target for _ in range(N * M)]
        V = [[0] * R_target for _ in range(M * P)]
        W = [[0] * R_target for _ in range(P * N)]

        # Initialize: use structured pattern from schoolbook, then perturb
        # Map each schoolbook term (i,j,k) to a column in the reduced rank
        # Use modulo to wrap around
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    r = col % R_target
                    U[i * M + j][r] += 1
                    V[j * P + k][r] += 1
                    W[k * N + i][r] += 1
                    col += 1

        # Add random perturbations
        n_perturb = R_target * 15
        for _ in range(n_perturb):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randrange(N * M)
                r = random.randrange(R_target)
                U[idx][r] += random.choice([-1, 1])
            elif mat == 'V':
                idx = random.randrange(M * P)
                r = random.randrange(R_target)
                V[idx][r] += random.choice([-1, 1])
            else:
                idx = random.randrange(P * N)
                r = random.randrange(R_target)
                W[idx][r] += random.choice([-1, 1])

        # Run local search
        U, V, W, err = local_search_fast(U, V, W, R_target, max_steps=50000)

        if err < 1e-10:
            # Final verification with rounded values
            U_int = [[round(U[i][r] * 2) / 2 for r in range(R_target)] for i in range(N * M)]
            V_int = [[round(V[i][r] * 2) / 2 for r in range(R_target)] for i in range(M * P)]
            W_int = [[round(W[i][r] * 2) / 2 for r in range(R_target)] for i in range(P * N)]
            if verify(U_int, V_int, W_int, R_target):
                return U_int, V_int, W_int

        return None

    # Try ranks from 119 down
    for R in range(target_rank - 1, max(100, target_rank - 30), -1):
        if time.time() > deadline - 15:
            break

        for restart in range(3):
            if time.time() > deadline - 8:
                break
            result = try_reduce_rank(R, seed=None)
            if result is not None:
                U, V, W = result
                if verify(U, V, W, R):
                    return {"U": U, "V": V, "W": W}

    # Fallback: return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
