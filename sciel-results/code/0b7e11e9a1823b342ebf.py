"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _compute_col_contribs(n, m, p, U_arr, V_arr, W_arr):
    """Precompute per-column tensor contributions."""
    rank = U_arr.shape[1]
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    return col_contrib


def _try_reduce_rank(n, m, p, U, V, W, deadline, max_group=4):
    """
    Aggressively remove columns (singles, pairs, triples, quadruples) that are redundant.
    Uses incremental tensor updates for speed.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)
    
    target = np.ones((n, m, p))
    tol = 1e-9
    
    col_contrib = _compute_col_contribs(n, m, p, U_arr, V_arr, W_arr)
    T_current = col_contrib.sum(axis=0)
    
    # Helper to try removing a group of columns
    def try_remove(indices):
        nonlocal T_current, keep
        if time.time() > deadline:
            return False
        sub = np.array(indices)
        T_new = T_current - col_contrib[sub].sum(axis=0)
        if np.allclose(T_new, target, atol=tol):
            for idx in indices:
                keep[idx] = False
            T_current = T_new
            return True
        return False
    
    # Phase 1: single-column removals, repeated
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if try_remove([r]):
                changed = True
                break
    
    # Phase 2: pair-column removals, repeated
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept_indices = [r for r in range(rank) if keep[r]]
        for i in range(len(kept_indices)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(kept_indices)):
                r1 = kept_indices[i]
                r2 = kept_indices[j]
                if not keep[r1] or not keep[r2]:
                    continue
                if try_remove([r1, r2]):
                    changed = True
                    break
            if changed:
                break
    
    # Phase 3: triple-column removals
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept_indices = [r for r in range(rank) if keep[r]]
        nk = len(kept_indices)
        if nk < 3:
            break
        for combo in combinations(range(nk), 3):
            if time.time() > deadline:
                break
            indices = [kept_indices[c] for c in combo]
            if all(keep[idx] for idx in indices):
                if try_remove(indices):
                    changed = True
                    break
    
    # Phase 4: quadruple-column removals (if max_group >= 4)
    if max_group >= 4:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            kept_indices = [r for r in range(rank) if keep[r]]
            nk = len(kept_indices)
            if nk < 4:
                break
            for combo in combinations(range(nk), 4):
                if time.time() > deadline:
                    break
                indices = [kept_indices[c] for c in combo]
                if all(keep[idx] for idx in indices):
                    if try_remove(indices):
                        changed = True
                        break
    
    # Phase 5: another round of singles and pairs after larger groups
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if try_remove([r]):
                changed = True
                break
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept_indices = [r for r in range(rank) if keep[r]]
        for i in range(len(kept_indices)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(kept_indices)):
                r1 = kept_indices[i]
                r2 = kept_indices[j]
                if not keep[r1] or not keep[r2]:
                    continue
                if try_remove([r1, r2]):
                    changed = True
                    break
            if changed:
                break
    
    # Phase 6: final single pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if try_remove([r]):
                changed = True
                break
    
    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 products for A[0:2, 0:2] * B[0:2, k0:k1+1]
        
        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (a10 + a11) * B[0,k0]
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: a11 * (B[1,k0] - B[0,k0])
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (a00 + a01) * B[1,k1]
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[2][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[5][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[5][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    # Build Strassen-based rank-55 decomposition
    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        # Fallback to trivial
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        # Aggressive rank reduction with groups up to size 4
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline, max_group=4)

    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
