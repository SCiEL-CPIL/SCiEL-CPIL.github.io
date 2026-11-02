"""Matrix multiplication 6x6x6 - rank-minimal tensor decomposition search."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def check_decomposition(Um, Vm, Wm):
        """Fast check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        R = Um.shape[1]
        # Build the multiplication tensor indices
        # For each term r: contributes U[r]⊗V[r]⊗W[r] to the tensor
        # We need the (i,j,k) slice to equal 1
        # Efficient: for each (i,j), the vector over k is sum_r U[i*m+j,r]*W[k*n+i,r] * V[j*p+k,r]
        # Let's just do it directly but vectorized
        for i in range(n):
            for j in range(m):
                u_row = Um[i * m + j]  # shape (R,)
                # For each k, need sum_r u_row[r] * Vm[j*p+k, r] * Wm[k*n+i, r]
                # = sum_r u_row[r] * (Vm[j*p+k] * Wm[k*n+i])[r]
                # Build matrix M where M[k, r] = Vm[j*p+k, r] * Wm[k*n+i, r]
                # Vm[j*p : j*p+p, :] has shape (p, R)
                # Wm[:, i] gives shape (p*n, 1), we need rows k*n+i for k in range(p)
                v_block = Vm[j * p : j * p + p, :]  # (p, R)
                w_rows = np.array([Wm[k * n + i] for k in range(p)])  # (p, R)
                M = v_block * w_rows  # (p, R)
                results = M @ u_row  # (p,)
                if not np.allclose(results, 1.0, atol=1e-7):
                    return False
        return True

    def check_decomposition_fast(Um, Vm, Wm):
        """Even faster: build the full tensor and compare to ones tensor."""
        R = Um.shape[1]
        # The tensor T[i,j,k] = sum_r Um[i*m+j, r] * Vm[j*p+k, r] * Wm[k*n+i, r]
        # We can compute this as: for each r, outer product
        # T = sum_r Um_r ⊗ Vm_r ⊗ Wm_r (reshaped)
        # Um shape (n*m, R), Vm shape (m*p, R), Wm shape (p*n, R)
        # Reshape Um to (n, m, R), Vm to (m, p, R), Wm to (p, n, R)
        Um_3d = Um.reshape(n, m, R)  # Um[i, j, r]
        Vm_3d = Vm.reshape(m, p, R)  # Vm[j, k, r]
        Wm_3d = Wm.reshape(p, n, R)  # Wm[k, i, r]
        # T[i,j,k] = sum_r Um_3d[i,j,r] * Vm_3d[j,k,r] * Wm_3d[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', Um_3d, Vm_3d, Wm_3d)
        if not np.allclose(T, 1.0, atol=1e-7):
            return False
        return True

    # Schoolbook initialization
    rank0 = n * m * p  # 216
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank0

    def try_remove_columns(Um, Vm, Wm, col_indices_to_try):
        """Try to remove a subset of columns. Returns (success, new_U, new_V, new_W)."""
        R = Um.shape[1]
        # Keep indices not in col_indices_to_try
        keep = [r for r in range(R) if r not in col_indices_to_try]
        if not keep:
            return False, None, None, None
        U_new = Um[:, keep]
        V_new = Vm[:, keep]
        W_new = Wm[:, keep]
        if check_decomposition_fast(U_new, V_new, W_new):
            return True, U_new, V_new, W_new
        return False, None, None, None

    # Phase 1: Greedy single-column removal
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    current_rank = rank0

    # Shuffle column order for diversity
    col_order = list(range(rank0))
    random.shuffle(col_order)

    for idx in col_order:
        if elapsed() > deadline - 20:
            break
        if idx >= current_rank:
            continue
        success, U_new, V_new, W_new = try_remove_columns(U_cur, V_cur, W_cur, {idx})
        if success:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            current_rank -= 1
            if current_rank < best_rank:
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
                best_rank = current_rank
            # Re-shuffle remaining
            col_order = [c for c in col_order if c not in {idx} and c < current_rank]
            random.shuffle(col_order)

    # Phase 2: Try removing pairs of columns
    if elapsed() < deadline - 30:
        pairs = list(zip(col_order, col_order[1:]))
        random.shuffle(pairs)
        for a, b in pairs[:500]:
            if elapsed() > deadline - 20:
                break
            if a >= current_rank or b >= current_rank:
                continue
            if a == b:
                continue
            success, U_new, V_new, W_new = try_remove_columns(U_cur, V_cur, W_cur, {a, b})
            if success:
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                current_rank -= 2
                if current_rank < best_rank:
                    best_U = U_cur.copy()
                    best_V = V_cur.copy()
                    best_W = W_cur.copy()
                    best_rank = current_rank
                col_order = [c for c in col_order if c not in {a, b} and c < current_rank]
                random.shuffle(col_order)

    # Phase 3: Randomized restarts - try different initial orderings
    for restart in range(10):
        if elapsed() > deadline - 25:
            break
        # Start fresh from schoolbook
        U_r = U.copy()
        V_r = V.copy()
        W_r = W.copy()
        rank_r = rank0
        order = list(range(rank_r))
        random.shuffle(order)

        for idx in order:
            if elapsed() > deadline - 20:
                break
            if idx >= rank_r:
                continue
            success, U_new, V_new, W_new = try_remove_columns(U_r, V_r, W_r, {idx})
            if success:
                U_r, V_r, W_r = U_new, V_new, W_new
                rank_r -= 1
                if rank_r < best_rank:
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                    best_rank = rank_r
                order = [c for c in order if c != idx and c < rank_r]
                random.shuffle(order)

    # Final verification
    if not check_decomposition_fast(best_U, best_V, best_W):
        # Fallback
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank0

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
