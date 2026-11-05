import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]
DEADLINE = 145.0

def _strassen_2x2x2_embed(n, m, p, i0, j0, k0):
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # M2: (A10+A11)B00 -> C10 - C11
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # M3: A00(B01-B11) -> C01 + C11
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # M4: A11(B10-B00) -> C00 + C10
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # M5: (A00+A01)B11 -> -C00 + C01
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk):
    rank = bi * bj * bk
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(bi):
        for j in range(bj):
            for k in range(bk):
                U[(i0 + i) * m + (j0 + j), col] = 1
                V[(j0 + j) * p + (k0 + k), col] = 1
                W[(k0 + k) * n + (i0 + i), col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    if rank == 0:
        return False
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _verify_fast(U, V, W, n, m, p, cached_T=None):
    """Faster verification using sparse structure. Returns (bool, T_array)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    if rank == 0:
        return False, None
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9), T

def _try_eliminate_columns(U, V, W, n, m, p, max_time=15.0):
    """Try to find and eliminate linearly dependent columns using iterative greedy + pair checks."""
    t_start = time.time()
    rank = U.shape[1]
    if rank < 2:
        return U, V, W
    
    U_list = [U[:, i] for i in range(rank)]
    V_list = [V[:, i] for i in range(rank)]
    W_list = [W[:, i] for i in range(rank)]
    
    kept = list(range(rank))
    improved = True
    
    while improved and (time.time() - t_start) < max_time:
        improved = False
        for trial in range(len(kept)):
            if time.time() - t_start > max_time:
                break
            for idx in range(len(kept)):
                col = kept[idx]
                new_kept = kept[:idx] + kept[idx+1:]
                if len(new_kept) == 0:
                    continue
                
                U_new = np.column_stack([U_list[c] for c in new_kept])
                V_new = np.column_stack([V_list[c] for c in new_kept])
                W_new = np.column_stack([W_list[c] for c in new_kept])
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    kept = new_kept
                    improved = True
                    break
    
    if len(kept) == rank:
        return U, V, W
    
    U_new = np.column_stack([U_list[c] for c in kept])
    V_new = np.column_stack([V_list[c] for c in kept])
    W_new = np.column_stack([W_list[c] for c in kept])
    
    return U_new, V_new, W_new

def _try_pair_elimination(U, V, W, n, m, p, max_time=20.0):
    """Try eliminating pairs of columns that might be linearly combinable."""
    t_start = time.time()
    rank = U.shape[1]
    if rank < 4:
        return U, V, W
    
    U_list = [U[:, i] for i in range(rank)]
    V_list = [V[:, i] for i in range(rank)]
    W_list = [W[:, i] for i in range(rank)]
    
    kept = list(range(rank))
    improved = True
    
    while improved and (time.time() - t_start) < max_time:
        improved = False
        for a in range(len(kept)):
            if time.time() - t_start > max_time:
                break
            for b in range(a + 1, len(kept)):
                col_a = kept[a]
                col_b = kept[b]
                new_kept = kept[:a] + kept[a+1:b] + kept[b+1:]
                if len(new_kept) == 0:
                    continue
                
                U_new = np.column_stack([U_list[c] for c in new_kept])
                V_new = np.column_stack([V_list[c] for c in new_kept])
                W_new = np.column_stack([W_list[c] for c in new_kept])
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    kept = new_kept
                    improved = True
                    break
            if improved:
                break
    
    if len(kept) == rank:
        return U, V, W
    
    U_new = np.column_stack([U_list[c] for c in kept])
    V_new = np.column_stack([V_list[c] for c in kept])
    W_new = np.column_stack([W_list[c] for c in kept])
    
    return U_new, V_new, W_new

def _build_decomposition(n, m, p, i_groups, j_groups, k_groups):
    """Build decomposition from given groupings."""
    U_parts = []
    V_parts = []
    W_parts = []
    
    for i_start, i_size in i_groups:
        for j_start, j_size in j_groups:
            for k_start, k_size in k_groups:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, i_start, j_start, k_start)
                else:
                    U_s, V_s, W_s = _schoolbook_embed(n, m, p, i_start, j_start, k_start, i_size, j_size, k_size)
                U_parts.append(U_s)
                V_parts.append(V_s)
                W_parts.append(W_s)
    
    if not U_parts:
        return None, None, None
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    
    # Try multiple partitionings and pick the one with lowest rank after elimination
    # Partitioning options for dimension 4: [(0,2),(2,2)] or [(0,4)]
    # Partitioning options for dimension 5: various splits to maximize 2x2x2 Strassen blocks
    
    i_options = [
        [(0, 2), (2, 2)],
        [(0, 4)],
    ]
    
    j_options = [
        [(0, 2), (2, 2), (4, 1)],
        [(0, 1), (1, 2), (3, 2)],
        [(0, 2), (2, 1), (3, 2)],
        [(0, 3), (3, 2)],
        [(0, 2), (2, 3)],
        [(0, 5)],
    ]
    
    k_options = [
        [(0, 2), (2, 2), (4, 1)],
        [(0, 1), (1, 2), (3, 2)],
        [(0, 2), (2, 1), (3, 2)],
        [(0, 3), (3, 2)],
        [(0, 2), (2, 3)],
        [(0, 5)],
    ]
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for i_groups in i_options:
        for j_groups in j_options:
            for k_groups in k_options:
                if time.time() - t_start > DEADLINE - 30:
                    break
                
                U, V, W = _build_decomposition(n, m, p, i_groups, j_groups, k_groups)
                if U is None:
                    continue
                
                if not _verify(U, V, W, n, m, p):
                    continue
                
                # Try elimination
                remaining_time = DEADLINE - (time.time() - t_start) - 10
                if remaining_time < 5:
                    # Just use as-is
                    if U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    continue
                
                # Single column elimination
                elim_time = min(10.0, remaining_time * 0.4)
                U_e, V_e, W_e = _try_eliminate_columns(U, V, W, n, m, p, max_time=elim_time)
                if _verify(U_e, V_e, W_e, n, m, p) and U_e.shape[1] < U.shape[1]:
                    U, V, W = U_e, V_e, W_e
                
                # Pair elimination
                remaining_time = DEADLINE - (time.time() - t_start) - 10
                if remaining_time > 5:
                    pair_time = min(10.0, remaining_time * 0.5)
                    U_p, V_p, W_p = _try_pair_elimination(U, V, W, n, m, p, max_time=pair_time)
                    if _verify(U_p, V_p, W_p, n, m, p) and U_p.shape[1] < U.shape[1]:
                        U, V, W = U_p, V_p, W_p
                
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: pure schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
