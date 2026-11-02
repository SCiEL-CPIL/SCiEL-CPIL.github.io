import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under 150s

    # --- Build rank-28 Strassen baseline (verified) ---
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 8 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 8 x 28

        col = 0
        for b_blk in range(2):      # B column block: 0 or 1
            for a_blk in range(2):  # A col block / B row block: 0 or 1
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    k0 = 2 * b_blk
                    k1 = 2 * b_blk + 1

                    # U: A[i, j] -> flat index i*4 + j
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j][c] += coeff

                    # V: B[j, k] -> flat index j*4 + k
                    V_contrib = {
                        0: [(j0, k0, 1), (j1, k1, 1)],
                        1: [(j0, k0, 1)],
                        2: [(j0, k1, 1), (j1, k1, -1)],
                        3: [(j1, k0, 1), (j0, k0, -1)],
                        4: [(j1, k1, 1)],
                        5: [(j0, k0, 1), (j0, k1, 1)],
                        6: [(j1, k0, 1), (j1, k1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k][c] += coeff

                    # W: C[i, k] -> flat index k*2 + i
                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(k_loc + 2 * b_blk) * 2 + i_loc][c] += coeff

                    col += 1

        return {"U": U, "V": V, "W": W}, 28

    def validate(U, V, W, rank):
        """Check exact decomposition."""
        # Non-zero entries: for each (i,j,k) with matching indices, sum should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Zero entries: all other combinations should sum to 0
        for a in range(n*m):
            i1, j1 = a // m, a % m
            for b_idx in range(m*p):
                j2, k1 = b_idx // p, b_idx % p
                for c_idx in range(p*n):
                    k2, i2 = c_idx // n, c_idx % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0.0
                    for c in range(rank):
                        s += U[a][c] * V[b_idx][c] * W[c_idx][c]
                    if abs(s) > 1e-9:
                        return False
        return True

    # Build baseline
    baseline, base_rank = build_strassen_28()
    best_result = baseline
    best_rank = base_rank

    # --- Attempt rank reduction: replace off-diagonal Strassen blocks with rank-5 schemes ---
    # The four blocks are:
    #   Block 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (diagonal)
    #   Block 1: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  (off-diagonal: A cols 2-3, B rows 2-3)
    #   Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (off-diagonal: A cols 0-1, B rows 0-1)
    #   Block 3: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]  (diagonal)
    #
    # Off-diagonal blocks: A is 2x2, B is 2x2, but the multiplication is C = A * B where
    # A uses columns 2-3 (or 0-1) and B uses rows 2-3 (or 0-1).
    # Actually all four are 2x2 * 2x2 -> 2x2, so all have the same tensor structure.
    # The rank of 2x2*2x2 is 7 (Strassen is optimal for 2x2).
    # So we cannot reduce any individual 2x2 block below rank 7.
    #
    # Alternative: exploit the full 2x4 x 4x4 structure to share factors across blocks.
    # This is harder. Let's try a direct rank-27 search by perturbing the Strassen-28.

    # Try to find rank-27 by checking if any column of the Strassen-28 is redundant
    # after a linear transformation. This is equivalent to checking if the 28 rank-1
    # tensors span a 27-dimensional subspace that still contains the MM tensor.
    #
    # Simpler approach: try removing each column and checking if the remaining 27
    # columns can be linearly adjusted to still represent the tensor.
    # This is a nonlinear feasibility problem, so we use a heuristic:
    # For each column to remove, solve a least-squares problem for the adjustment.

    U28 = np.array(baseline["U"], dtype=float)  # 8 x 28
    V28 = np.array(baseline["V"], dtype=float)  # 16 x 28
    W28 = np.array(baseline["W"], dtype=float)  # 8 x 28

    # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to find rank-27: U', V', W' with 27 columns such that T is preserved.
    #
    # Approach: For each pair of columns (c1, c2) to merge, check if there exists
    # a scalar alpha such that replacing column c1 with (U[:,c1] + alpha*U[:,c2]) etc.
    # preserves the tensor. This is a rank-1 update condition.
    #
    # Actually, a cleaner approach: the tensor T can be written as a sum of rank-1 terms.
    # If two rank-1 terms u1 v1^T w1^T and u2 v2^T w2^T satisfy a linear dependency
    # in the tensor space, we might merge them. But this is complex.
    #
    # Let's try a more direct approach: use the fact that the MM tensor for 2x4x4
    # has a known structure and try to find a rank-27 decomposition via
    # structured random search with the Strassen-28 as initialization.

    # Attempt: for each column to remove, try to find a rank-1 correction
    # that compensates for the removal.
    for remove_col in range(28):
        if time.time() > deadline - 5:
            break

        # Remove column remove_col
        keep = [c for c in range(28) if c != remove_col]
        U_rem = U28[:, keep]  # 8 x 27
        V_rem = V28[:, keep]  # 16 x 27
        W_rem = W28[:, keep]  # 8 x 27

        # The removed column contributes: U28[:,remove_col] * V28[:,remove_col]^T * W28[:,remove_col]^T
        # as a rank-1 tensor. We need to see if this can be absorbed into the remaining 27 columns.
        #
        # This is equivalent to: can we find vectors u', v', w' (not necessarily unit) such that
        # the tensor T = sum_{c in keep} U_rem[:,c] V_rem[:,c]^T W_rem[:,c]^T + 
        #                  U28[:,remove_col] V28[:,remove_col]^T W28[:,remove_col]^T
        # can be written as a sum of 27 rank-1 terms?
        #
        # This is generally not possible unless the removed rank-1 term is in the span
        # of the other 27 rank-1 terms in the tensor space.
        #
        # Check: is the removed tensor in the span of the other 27?
        # Flatten the tensor: T is 2x4x4 = 32 elements.
        # The 27 remaining rank-1 tensors span at most 27 dimensions in the 32-dim space.
        # The removed tensor is 1 vector. If it's in their span, we can merge.

        # Build the 32-dim vector for the removed tensor
        def tensor_to_vec(u, v, w):
            """u: (2*4,), v: (4*4,), w: (4*2,) -> 32-dim vector for T[i,j,k]"""
            vec = np.zeros(32)
            for i in range(2):
                for j in range(4):
                    for k in range(4):
                        vec[i*16 + j*4 + k] = u[i*4+j] * v[j*4+k] * w[k*2+i]
            return vec

        removed_vec = tensor_to_vec(U28[:, remove_col], V28[:, remove_col], W28[:, remove_col])

        # Build the matrix of the 27 remaining rank-1 tensors
        M = np.zeros((32, 27))
        for idx, c in enumerate(keep):
            M[:, idx] = tensor_to_vec(U28[:, c], V28[:, c], W28[:, c])

        # Check if removed_vec is in the column space of M
        # Use SVD to check rank
        try:
            U_svd, S_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
            # Project removed_vec onto the column space
            proj = U_svd @ (U_svd.T @ removed_vec)
            residual = removed_vec - proj
            if np.linalg.norm(residual) < 1e-8:
                # The removed tensor is in the span! We can merge.
                # Find the coefficients
                coeffs = Vt_svd.T @ (Vt_svd @ removed_vec)  # This gives coords in V basis
                # Actually, solve M @ x = removed_vec
                x, residuals, rank_sv, sv = np.linalg.lstsq(M, removed_vec, rcond=None)
                if residuals[0] < 1e-8:
                    # We can absorb the removed column into the others.
                    # New decomposition: for each remaining column c, add x[idx_c] * removed_col
                    U_new = U_rem.copy()
                    V_new = V_rem.copy()
                    W_new = W_rem.copy()
                    for idx, c in enumerate(keep):
                        U_new[:, idx] += x[idx] * U28[:, remove_col]
                        V_new[:, idx] += x[idx] * V28[:, remove_col]
                        W_new[:, idx] += x[idx] * W28[:, remove_col]

                    # Verify
                    U_list = U_new.tolist()
                    V_list = V_new.tolist()
                    W_list = W_new.tolist()
                    if validate(U_list, V_list, W_list, 27):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = 27
                        break
        except np.linalg.LinAlgError:
            pass

    # If rank 27 not found via merging, try a structured search for rank 27
    # using the Strassen-28 as a starting point with small perturbations.
    if best_rank > 27 and time.time() < deadline - 10:
        import random
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            random.seed(12345 + attempt)

            # Start from Strassen-28, randomly zero one column, then perturb
            remove_col = random.randint(0, 27)
            keep = [c for c in range(28) if c != remove_col]
            U = [row[:remove_col] + row[remove_col+1:] for row in baseline["U"]]
            V = [row[:remove_col] + row[remove_col+1:] for row in baseline["V"]]
            W = [row[:remove_col] + row[remove_col+1:] for row in baseline["W"]]

            # Small perturbations
            for _ in range(10):
                # Randomly adjust a few entries
                for _ in range(5):
                    mat = random.choice([U, V, W])
                    r = random.randint(0, len(mat)-1)
                    c = random.randint(0, 26)
                    delta = random.choice([-0.5, -1.0, 0.5, 1.0])
                    mat[r][c] += delta

            if validate(U, V, W, 27):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = 27
                break

    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
