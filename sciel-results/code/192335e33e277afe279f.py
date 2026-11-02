import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Build rank-50 decomposition (Strassen on 2x2 blocks + handling)
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1

        U[u_c, col+1] += 1
        U[u_d, col+1] += 1
        V[v_e, col+1] += 1

        U[u_a, col+2] += 1
        V[v_f, col+2] += 1
        V[v_h, col+2] -= 1

        U[u_d, col+3] += 1
        V[v_g, col+3] += 1
        V[v_e, col+3] -= 1

        U[u_a, col+4] += 1
        U[u_b, col+4] += 1
        V[v_h, col+4] += 1

        U[u_c, col+5] += 1
        U[u_a, col+5] -= 1
        V[v_e, col+5] += 1
        V[v_f, col+5] += 1

        U[u_b, col+6] += 1
        U[u_d, col+6] -= 1
        V[v_g, col+6] += 1
        V[v_h, col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00, col] += 1
        W[w_00, col+3] += 1
        W[w_00, col+4] -= 1
        W[w_00, col+6] += 1

        W[w_01, col+2] += 1
        W[w_01, col+4] += 1

        W[w_10, col+1] += 1
        W[w_10, col+3] += 1

        W[w_11, col] += 1
        W[w_11, col+1] -= 1
        W[w_11, col+2] += 1
        W[w_11, col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a, col] += 1
    V[v_e, col] += 1
    U[u_b, col+1] += 1
    V[v_g, col+1] += 1
    U[u_c, col+2] += 1
    V[v_e, col+2] += 1
    U[u_d, col+3] += 1
    V[v_g, col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08, col] += 1
    W[w_08, col+1] += 1
    W[w_18, col+2] += 1
    W[w_18, col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0, col] += 1
        V[v_b, col] += 1
        W[k * n + 0, col] += 1
        U[u_ar1, col+1] += 1
        V[v_b, col+1] += 1
        W[k * n + 1, col+1] += 1
        col += 2

    # Verify initial decomposition
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # === AGGRESSIVE RANK REDUCTION ===
    
    # Step 1: Remove duplicate columns
    unique_indices = []
    seen = {}
    for r in range(rank):
        key = (tuple(U[:, r].tolist()), tuple(V[:, r].tolist()), tuple(W[:, r].tolist()))
        if key not in seen:
            seen[key] = r
            unique_indices.append(r)
    
    if len(unique_indices) < rank:
        U = U[:, unique_indices]
        V = V[:, unique_indices]
        W = W[:, unique_indices]
        rank = len(unique_indices)

    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    best_rank = current_rank

    def verify_decomp(Ur, Vr, Wr, T_target):
        """Check if decomposition is exact."""
        d = np.einsum('ir,jr,kr->ijk', Ur, Vr, Wr)
        return np.allclose(d, T_target, atol=1e-9)

    def try_remove_column(Ur, Vr, Wr, col_idx, T_target):
        """Try removing a single column."""
        mask = np.ones(Ur.shape[1], dtype=bool)
        mask[col_idx] = False
        return verify_decomp(Ur[:, mask], Vr[:, mask], Wr[:, mask], T_target)

    def try_remove_two_columns(Ur, Vr, Wr, i, j, T_target):
        """Try removing two columns."""
        mask = np.ones(Ur.shape[1], dtype=bool)
        mask[i] = False
        mask[j] = False
        return verify_decomp(Ur[:, mask], Vr[:, mask], Wr[:, mask], T_target)

    # Step 2: Iterative single-column removal
    removed = True
    while removed and time.time() < deadline - 30:
        removed = False
        for r in range(current_rank):
            if time.time() > deadline - 30:
                break
            if try_remove_column(current_U, current_V, current_W, r, T):
                mask = np.ones(current_rank, dtype=bool)
                mask[r] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                removed = True
                break  # restart loop since indices changed
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Step 3: Try pairwise removal (more expensive but can find dependencies)
    if time.time() < deadline - 20:
        for i in range(current_rank):
            if time.time() > deadline - 20:
                break
            for j in range(i + 1, current_rank):
                if time.time() > deadline - 20:
                    break
                if try_remove_two_columns(current_U, current_V, current_W, i, j, T):
                    mask = np.ones(current_rank, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_rank -= 2
                    break
            if time.time() > deadline - 20:
                break
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Step 4: Null-space based approach for further reduction
    if time.time() < deadline - 15:
        # Build coefficient matrix: each column r maps to vec_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        n_entries = n * m * m * p * p * n
        mat = np.zeros((n_entries, current_rank))
        for r in range(current_rank):
            vec = np.einsum('i,j,k->ijk', current_U[:, r], current_V[:, r], current_W[:, r]).flatten()
            mat[:, r] = vec

        try:
            _, s, Vt = np.linalg.svd(mat, full_matrices=False)
            tol = s[0] * current_rank * 1e-10
            num_rank = int(np.sum(s > tol))
            null_dim = current_rank - num_rank

            if null_dim > 0:
                # Get null space vectors
                null_vectors = Vt[num_rank:]

                # Try to find a combination of columns that can be removed
                # For each null vector, find the column with largest coefficient
                for nv_idx in range(min(null_dim, 10)):
                    if time.time() > deadline - 15:
                        break
                    null_vec = null_vectors[nv_idx]
                    abs_null = np.abs(null_vec)
                    max_val = np.max(abs_null)
                    if max_val < 1e-10:
                        continue
                    # Try removing the column with the largest null-space coefficient
                    drop_col = int(np.argmax(abs_null))
                    if try_remove_column(current_U, current_V, current_W, drop_col, T):
                        mask = np.ones(current_rank, dtype=bool)
                        mask[drop_col] = False
                        current_U = current_U[:, mask]
                        current_V = current_V[:, mask]
                        current_W = current_W[:, mask]
                        current_rank -= 1
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        except Exception:
            pass

    # Step 5: Final attempt - try all single removals again after reductions
    if time.time() < deadline - 10:
        removed = True
        while removed and time.time() < deadline - 10:
            removed = False
            for r in range(current_rank):
                if time.time() > deadline - 10:
                    break
                if try_remove_column(current_U, current_V, current_W, r, T):
                    mask = np.ones(current_rank, dtype=bool)
                    mask[r] = False
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_rank -= 1
                    removed = True
                    break
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
