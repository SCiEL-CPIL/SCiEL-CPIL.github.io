import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 5, 5]

def _build_decomposition(n, m, p):
    """Build block decomposition with Strassen 2x2x2 where possible."""
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_block(i_off, j_off, k_off, bn, bm, bp):
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    ig = i_off + i
                    jg = j_off + j
                    kg = k_off + k
                    add_term(
                        {ig * m + jg: 1},
                        {jg * p + kg: 1},
                        {kg * n + ig: 1}
                    )
    
    def strassen_block(i_off, j_off, k_off):
        def gi(i, j): return (i_off + i) * m + (j_off + j)
        def gj(j, k): return (j_off + j) * p + (k_off + k)
        def gk(k, i): return (k_off + k) * n + (i_off + i)
        
        # Strassen 7 terms for 2x2x2
        add_term(
            {gi(0,0): 1, gi(1,1): 1},
            {gj(0,0): 1, gj(1,1): 1},
            {gk(0,0): 1, gk(1,1): 1}
        )
        add_term(
            {gi(1,0): 1, gi(1,1): 1},
            {gj(0,0): 1},
            {gk(0,1): 1, gk(1,1): -1}
        )
        add_term(
            {gi(0,0): 1},
            {gj(0,1): 1, gj(1,1): -1},
            {gk(1,0): 1, gk(1,1): 1}
        )
        add_term(
            {gi(1,1): 1},
            {gj(1,0): 1, gj(0,0): -1},
            {gk(0,0): 1, gk(0,1): 1}
        )
        add_term(
            {gi(0,0): 1, gi(0,1): 1},
            {gj(1,1): 1},
            {gk(0,0): -1, gk(1,0): 1}
        )
        add_term(
            {gi(1,0): 1, gi(0,0): -1},
            {gj(0,0): 1, gj(0,1): 1},
            {gk(1,1): 1}
        )
        add_term(
            {gi(0,1): 1, gi(1,1): -1},
            {gj(1,0): 1, gj(1,1): 1},
            {gk(0,0): 1}
        )
    
    # Partition: maximize 2x2x2 Strassen blocks
    i_parts = [(0, 2), (2, 2)]
    j_parts = [(0, 2), (2, 2), (4, 1)]
    k_parts = [(0, 2), (2, 2), (4, 1)]
    
    for i_off, i_sz in i_parts:
        for j_off, j_sz in j_parts:
            for k_off, k_sz in k_parts:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    strassen_block(i_off, j_off, k_off)
                else:
                    schoolbook_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    return terms

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    # Build tensor contribution
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape and use einsum
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _reduce_rank_linear(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """Use linear algebra to find minimal subset of terms."""
    rank = U.shape[1]
    total_entries = n * m * p
    
    # Build the 100 x rank matrix M where M[t,c] is the contribution of term c to entry t
    M = np.zeros((total_entries, rank))
    for c in range(rank):
        u_vals = U[:, c]
        v_vals = V[:, c]
        w_vals = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    M[t, c] = u_vals[i * m + j] * v_vals[j * p + k] * w_vals[k * n + i]
    
    target = np.ones(total_entries)
    
    # Check current validity
    current_sum = M @ np.ones(rank)
    if not np.allclose(current_sum, target, atol=tol):
        return U, V, W
    
    # Try to find minimal subset using greedy + linear algebra
    # First, find the rank of M
    M_rank = np.linalg.matrix_rank(M, tol=tol)
    
    # Greedy: try removing terms one by one, but use linear algebra to skip
    kept = list(range(rank))
    
    # Multiple passes
    for pass_num in range(3):
        if deadline and time.time() > deadline - 5:
            break
        changed = False
        for idx in range(len(kept) - 1, -1, -1):
            if deadline and time.time() > deadline - 5:
                break
            trial = kept[idx]
            new_kept = kept[:idx] + kept[idx+1:]
            if not new_kept:
                continue
            # Check if removing this term still gives correct sum
            # M[:, new_kept] @ ones should equal target
            if len(new_kept) < total_entries:
                # Quick check: sum of remaining columns
                s = M[:, new_kept].sum(axis=1)
            else:
                s = M[:, new_kept] @ np.ones(len(new_kept))
            if np.allclose(s, target, atol=tol):
                kept = new_kept
                changed = True
        if not changed:
            break
    
    if len(kept) < rank:
        return U[:, kept], V[:, kept], W[:, kept]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    terms = _build_decomposition(n, m, p)
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Reduce rank
    U, V, W = _reduce_rank_linear(U, V, W, n, m, p, deadline=deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
