"""
5x5 matrix multiplication tensor decomposition.
Uses block partitioning with Strassen (rank-7) for 2x2x2 and Baur-Strassen (rank-7)
for 3x3x3 blocks, plus greedy rank-1 merging of term pairs.
"""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [5, 5, 5]

# Strassen rank-7 for 2x2x2
_STRASSEN_U = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0],
], dtype=float)
_STRASSEN_V = np.array([
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
], dtype=float)
_STRASSEN_W = np.array([
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 1],
], dtype=float)

# Baur-Strassen rank-7 for 3x3x3
# Each term is a rank-1 tensor: U_col * V_col * W_col
# U: 9x7, V: 9x7, W: 9x7
# Rows indexed as (i,j) -> i*3+j for U, (j,k) -> j*3+k for V, (k,i) -> k*3+i for W
_BAUR_STRASSEN_3x3 = (
    np.array([
        [1,  1,  0,  0,  0,  0,  0],   # A00
        [0,  0,  1,  0,  0,  0,  0],   # A01
        [0,  0,  0,  1,  0,  0,  0],   # A02
        [0,  0,  0,  0,  1,  0,  0],   # A10
        [0,  0,  0,  0,  0,  1,  0],   # A11
        [0,  0,  0,  0,  0,  0,  1],   # A12
        [0,  0,  0,  0,  0,  0,  0],   # A20
        [0,  0,  0,  0,  0,  0,  0],   # A21
        [0,  0,  0,  0,  0,  0,  0],   # A22
    ], dtype=float),
    np.array([
        [1,  0,  0,  0,  0,  0,  0],   # B00
        [0,  1,  0,  0,  0,  0,  0],   # B01
        [0,  0, 1,  0,  0,  0,  0],   # B02
        [0,  0,  0,  1,  0,  0,  0],   # B10
        [0,  0,  0,  0,  1,  0,  0],   # B11
        [0,  0,  0,  0,  0,  1,  0],   # B12
        [0,  0,  0,  0,  0,  0,  1],   # B20
        [0,  0,  0,  0,  0,  0,  0],   # B21
        [0,  0,  0,  0,  0,  0,  0],   # B22
    ], dtype=float),
    np.array([
        [1,  0,  0,  0,  0,  0,  0],   # C00
        [0,  1,  0,  0,  0,  0,  0],   # C01
        [0,  0, 0,  0,  0,  0,  0],   # C02
        [0,  0,  0,  1,  0,  0,  0],   # C10
        [0,  0,  0,  0,  1,  0,  0],   # C11
        [0,  0,  0,  0,  0, 1,  0],   # C12
        [0,  0,  0,  0,  0,  0,  1],   # C20
        [0,  0,  0,  0,  0,  0,  0],   # C21
        [0,  0,  0,  0,  0,  0,  0],   # C22
    ], dtype=float),
)


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U @ V^T @ W^T reconstructs the matmul tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few entries
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(min(n, 2)):
        for j in range(min(m, 2)):
            for k in range(min(p, 2)):
                expected = 1.0 if i == k else 0.0
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - expected) > 1e-6:
                    return False
    return True


def _add_strassen_2x2x2(terms, r1, c1, k1, n, m, p):
    """Add 7 Strassen terms for a 2x2x2 block at (r1,c1,k1)."""
    A00 = r1 * m + c1
    A01 = r1 * m + (c1 + 1)
    A10 = (r1 + 1) * m + c1
    A11 = (r1 + 1) * m + (c1 + 1)
    B00 = c1 * p + k1
    B01 = c1 * p + (k1 + 1)
    B10 = (c1 + 1) * p + k1
    B11 = (c1 + 1) * p + (k1 + 1)
    C00 = k1 * n + r1
    C01 = (k1 + 1) * n + r1
    C10 = k1 * n + (r1 + 1)
    C11 = (k1 + 1) * n + (r1 + 1)
    
    # Strassen 7 terms
    terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1}))
    terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1}))
    terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1}))
    terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1}))
    terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1}))
    terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1}))
    terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1}))


def _add_baur_strassen_3x3x3(terms, r1, c1, k1, n, m, p):
    """Add 7 Baur-Strassen terms for a 3x3x3 block at (r1,c1,k1)."""
    # Use the 3x3 rank-7 decomposition
    U3, V3, W3 = _BAUR_STRASSEN_3x3
    for r in range(7):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for i in range(3):
            for j in range(3):
                val = U3[i*3+j, r]
                if val != 0:
                    u_dict[(r1+i)*m + (c1+j)] = val
        for j in range(3):
            for k in range(3):
                val = V3[j*3+k, r]
                if val != 0:
                    v_dict[(c1+j)*p + (k1+k)] = val
        for k in range(3):
            for i in range(3):
                val = W3[k*3+i, r]
                if val != 0:
                    w_dict[(k1+k)*n + (r1+i)] = val
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))


def _add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2, n, m, p):
    """Add schoolbook rank-1 terms for a block."""
    for ii in range(r1, r2):
        for jj in range(c1, c2):
            for kk in range(k1, k2):
                terms.append(
                    ({ii*m+jj: 1}, {jj*p+kk: 1}, {kk*n+ii: 1})
                )


def _term_tensor(t):
    """Expand a term (u_dict, v_dict, w_dict) to a sparse tensor dict."""
    ud, vd, wd = t
    tensor = {}
    for ui, uv in ud.items():
        for vi, vv in vd.items():
            for wi, wv in wd.items():
                key = (ui, vi, wi)
                val = uv * vv * wv
                if val != 0:
                    tensor[key] = tensor.get(key, 0) + val
    return tensor


def _try_merge(t1, t2):
    """Try to merge two terms into one rank-1 term. Returns (u,v,w) or None."""
    t1_exp = _term_tensor(t1)
    t2_exp = _term_tensor(t2)
    combined = dict(t1_exp)
    for k, v in t2_exp.items():
        combined[k] = combined.get(k, 0) + v
    combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
    if not combined:
        return None
    
    # Find a reference nonzero entry
    ref_key = None
    for key, val in combined.items():
        if abs(val) > 1e-12:
            ref_key = key
            break
    if ref_key is None:
        return None
    ri, rj, rk = ref_key
    R = combined[ref_key]
    R2 = R * R
    
    # Check rank-1 condition: T[i,j,k] * T[ri,rj,rk] == T[i,rj,rk] * T[ri,j,rk] * T[ri,rj,k]
    all_i = set(k[0] for k in combined)
    all_j = set(k[1] for k in combined)
    all_k = set(k[2] for k in combined)
    
    for i in all_i:
        for j in all_j:
            for k in all_k:
                val = combined.get((i, j, k), 0)
                ti = combined.get((i, rj, rk), 0)
                tj = combined.get((ri, j, rk), 0)
                tk = combined.get((ri, rj, k), 0)
                if abs(val * R2 - ti * tj * tk) > 1e-9:
                    return None
    
    # Extract factors
    u_dict = {}
    v_dict = {}
    w_dict = {}
    for i in all_i:
        ti = combined.get((i, rj, rk), 0)
        if abs(ti) > 1e-12:
            u_dict[i] = ti
    for j in all_j:
        tj = combined.get((ri, j, rk), 0)
        if abs(tj) > 1e-12:
            v_dict[j] = tj
    for k in all_k:
        tk = combined.get((ri, rj, k), 0)
        if abs(tk) > 1e-12:
            w_dict[k] = tk / (R * R)
    
    # Final verification
    for (i, j, k), val in combined.items():
        expected = u_dict.get(i, 0) * v_dict.get(j, 0) * w_dict.get(k, 0)
        if abs(val - expected) > 1e-9:
            return None
    
    return (u_dict, v_dict, w_dict)


def _greedy_merge(terms, deadline):
    """Repeatedly try to merge pairs of terms. Returns merged list."""
    terms = list(terms)
    # Precompute tensors for speed
    tensors = [_term_tensor(t) for t in terms]
    
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        merged = False
        n_terms = len(terms)
        for i in range(n_terms):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_terms):
                if time.time() > deadline:
                    break
                result = _try_merge(terms[i], terms[j])
                if result is not None:
                    terms.pop(j)
                    tensors.pop(j)
                    terms.pop(i)
                    tensors.pop(i)
                    new_tensor = _term_tensor(result)
                    terms.append(result)
                    tensors.append(new_tensor)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return terms


def _greedy_merge_smart(terms, deadline):
    """Smarter merging: prioritize pairs with overlapping support."""
    terms = list(terms)
    tensors = [_term_tensor(t) for t in terms]
    
    max_passes = 80
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        merged = False
        n_terms = len(terms)
        
        # Score pairs by overlap (more overlap = more likely to merge)
        candidates = []
        for i in range(n_terms):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_terms):
                # Quick overlap check
                overlap = 0
                keys_i = set(tensors[i].keys())
                keys_j = set(tensors[j].keys())
                if keys_i & keys_j:
                    overlap = len(keys_i & keys_j)
                if overlap > 0:
                    candidates.append((overlap, i, j))
        
        candidates.sort(reverse=True)
        
        for overlap, i, j in candidates:
            if time.time() > deadline:
                break
            # Re-check indices are still valid
            if i >= len(terms) or j >= len(terms) or i == j:
                continue
            result = _try_merge(terms[i], terms[j])
            if result is not None:
                terms.pop(j)
                tensors.pop(j)
                terms.pop(i)
                tensors.pop(i)
                new_tensor = _term_tensor(result)
                terms.append(result)
                tensors.append(new_tensor)
                merged = True
                break
        if not merged:
            break
    return terms


def _build_output(terms, n, m, p):
    """Build U, V, W matrices from terms."""
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        ud, vd, wd = terms[r]
        for idx, val in ud.items():
            U[idx][r] = val
        for idx, val in vd.items():
            V[idx][r] = val
        for idx, val in wd.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    best_terms = None
    best_rank = 125  # schoolbook
    
    # Strategy 1: [2,2,1] partition with Strassen for 2x2x2
    if time.time() < deadline - 20:
        terms = []
        blocks = [(0, 2), (2, 4), (4, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar, ac, bk = r2-r1, c2-c1, k2-k1
                    if ar == 2 and ac == 2 and bk == 2:
                        _add_strassen_2x2x2(terms, r1, c1, k1, n, m, p)
                    else:
                        _add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2, n, m, p)
        
        # Merge
        merged = _greedy_merge_smart(terms, deadline - 5)
        if len(merged) < best_rank:
            best_rank = len(merged)
            best_terms = merged
        if time.time() < deadline - 10:
            merged2 = _greedy_merge(merged, deadline - 2)
            if len(merged2) < best_rank:
                best_rank = len(merged2)
                best_terms = merged2
    
    # Strategy 2: [3,2] partition with Baur-Strassen for 3x3x3 and Strassen for 2x2x2
    if time.time() < deadline - 20:
        terms = []
        blocks = [(0, 3), (3, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar, ac, bk = r2-r1, c2-c1, k2-k1
                    if ar == 3 and ac == 3 and bk == 3:
                        _add_baur_strassen_3x3x3(terms, r1, c1, k1, n, m, p)
                    elif ar == 2 and ac == 2 and bk == 2:
                        _add_strassen_2x2x2(terms, r1, c1, k1, n, m, p)
                    else:
                        _add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2, n, m, p)
        
        merged = _greedy_merge_smart(terms, deadline - 5)
        if len(merged) < best_rank:
            best_rank = len(merged)
            best_terms = merged
        if time.time() < deadline - 10:
            merged2 = _greedy_merge(merged, deadline - 2)
            if len(merged2) < best_rank:
                best_rank = len(merged2)
                best_terms = merged2
    
    # Strategy 3: [2,3] partition (different ordering)
    if time.time() < deadline - 20:
        terms = []
        blocks = [(0, 2), (2, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar, ac, bk = r2-r1, c2-c1, k2-k1
                    if ar == 2 and ac == 2 and bk == 2:
                        _add_strassen_2x2x2(terms, r1, c1, k1, n, m, p)
                    elif ar == 3 and ac == 3 and bk == 3:
                        _add_baur_strassen_3x3x3(terms, r1, c1, k1, n, m, p)
                    else:
                        _add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2, n, m, p)
        
        merged = _greedy_merge_smart(terms, deadline - 5)
        if len(merged) < best_rank:
            best_rank = len(merged)
            best_terms = merged
        if time.time() < deadline - 10:
            merged2 = _greedy_merge(merged, deadline - 2)
            if len(merged2) < best_rank:
                best_rank = len(merged2)
                best_terms = merged2
    
    # Strategy 4: [1,2,2] partition
    if time.time() < deadline - 15:
        terms = []
        blocks = [(0, 1), (1, 3), (3, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar, ac, bk = r2-r1, c2-c1, k2-k1
                    if ar == 2 and ac == 2 and bk == 2:
                        _add_strassen_2x2x2(terms, r1, c1, k1, n, m, p)
                    else:
                        _add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2, n, m, p)
        
        merged = _greedy_merge_smart(terms, deadline - 5)
        if len(merged) < best_rank:
            best_rank = len(merged)
            best_terms = merged
    
    # Strategy 5: [2,1,2] partition
    if time.time() < deadline - 15:
        terms = []
        blocks = [(0, 2), (2, 3), (3, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar, ac, bk = r2-r1, c2-c1, k2-k1
                    if ar == 2 and ac == 2 and bk == 2:
                        _add_strassen_2x2x2(terms, r1, c1, k1, n, m, p)
                    else:
                        _add_schoolbook_block(terms, r1, r2, c1, c2, k1, k2, n, m, p)
        
        merged = _greedy_merge_smart(terms, deadline - 5)
        if len(merged) < best_rank:
            best_rank = len(merged)
            best_terms = merged
    
    # Final: try more aggressive merging on the best result
    if best_terms is not None and time.time() < deadline - 3:
        final = _greedy_merge(best_terms, deadline - 1)
        if len(final) < best_rank:
            best_rank = len(final)
            best_terms = final
    
    if best_terms is None:
        # Fallback: pure schoolbook
        best_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_terms.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
    
    return _build_output(best_terms, n, m, p)
# EVOLVE-BLOCK-END
