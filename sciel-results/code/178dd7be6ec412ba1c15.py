import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
# U shape: (n*m, r) = (9, r)
# V shape: (m*p, r) = (12, r)
# W shape: (p*n, r) = (12, r)

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    # Check half-integer constraint
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _strassen_2x2_block(i_base, j_base, k_base):
    """
    Returns the U, V, W vectors for a 2x2x2 Strassen block.
    """
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Term 1: (A00+A11)(B00+B11)
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    # Term 2: (A10+A11)B00
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    # Term 3: A00(B01-B11)
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    # Term 4: A11(B10-B00)
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    # Term 5: (A00+A01)B11
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    # Term 6: (A10-A00)(B00+B01)
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    # Term 7: (A01-A11)(B10+B11)
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

def _solve_remaining_greedy(remaining_terms):
    """
    Try to decompose the remaining terms with minimal rank.
    Uses a greedy approach to combine terms.
    """
    from collections import defaultdict
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    if not remaining_terms:
        return U_cols, V_cols, W_cols
    
    # Group by j (middle index)
    by_j = defaultdict(list)
    for (i, j, k) in remaining_terms:
        by_j[j].append((i, k))
    
    for j, terms in by_j.items():
        terms = sorted(terms)
        num_terms = len(terms)
        
        # Try to find the best pairing
        used = [False] * num_terms
        pairs = []
        
        for a in range(num_terms):
            if used[a]:
                continue
            for b in range(a+1, num_terms):
                if used[b]:
                    continue
                i1, k1 = terms[a]
                i2, k2 = terms[b]
                
                # Can combine if i1 != i2 and k1 != k2
                if i1 != i2 and k1 != k2:
                    pairs.append((a, b))
                    used[a] = True
                    used[b] = True
                    break
        
        # Create terms for pairs
        for a, b in pairs:
            i1, k1 = terms[a]
            i2, k2 = terms[b]
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            u_vec[i1 * m + j] = 1
            u_vec[i2 * m + j] = 1
            v_vec[j * p + k1] = 1
            v_vec[j * p + k2] = 1
            w_vec[k1 * n + i1] = 1
            w_vec[k2 * n + i2] = 1
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
        
        # Handle unpaired terms
        for a in range(num_terms):
            if not used[a]:
                i1, k1 = terms[a]
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_vec[i1 * m + j] = 1
                v_vec[j * p + k1] = 1
                w_vec[k1 * n + i1] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
    
    return U_cols, V_cols, W_cols

def _try_block_config(blocks):
    """
    Try a specific configuration of Strassen blocks and solve the remainder.
    Returns (rank, U, V, W) or None if not feasible.
    """
    strassen_covered = set()
    for ib, jb, kb in blocks:
        for i in range(ib, ib+2):
            for j in range(jb, jb+2):
                for k in range(kb, kb+2):
                    strassen_covered.add((i, j, k))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ib, jb, kb in blocks:
        U_terms, V_terms, W_terms = _strassen_2x2_block(ib, jb, kb)
        for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for idx, val in u_t.items():
                u_vec[idx] = val
            for idx, val in v_t.items():
                v_vec[idx] = val
            for idx, val in w_t.items():
                w_vec[idx] = val
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    # Collect remaining terms
    remaining = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    remaining.append((i, j, k))
    
    # Solve remaining
    rem_U, rem_V, rem_W = _solve_remaining_greedy(remaining)
    U_cols.extend(rem_U)
    V_cols.extend(rem_V)
    W_cols.extend(rem_W)
    
    if not U_cols:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    rank = U.shape[1]
    
    if _verify(U, V, W):
        return (rank, U, V, W)
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    best_result = None
    best_rank = float('inf')
    
    # Generate all possible block positions
    all_blocks = []
    for ib in range(0, n-1):
        for jb in range(0, m-1):
            for kb in range(0, p-1):
                all_blocks.append((ib, jb, kb))
    
    # Try different numbers of blocks
    # Try 0 blocks (schoolbook)
    result = _try_block_config([])
    if result and result[0] < best_rank:
        best_rank = result[0]
        best_result = result
    
    # Try 1 block
    for block in all_blocks:
        if time.time() > deadline:
            break
        result = _try_block_config([block])
        if result and result[0] < best_rank:
            best_rank = result[0]
            best_result = result
    
    # Try 2 blocks
    for combo in combinations(all_blocks, 2):
        if time.time() > deadline:
            break
        result = _try_block_config(list(combo))
        if result and result[0] < best_rank:
            best_rank = result[0]
            best_result = result
    
    # Try 3 blocks
    for combo in combinations(all_blocks, 3):
        if time.time() > deadline:
            break
        result = _try_block_config(list(combo))
        if result and result[0] < best_rank:
            best_rank = result[0]
            best_result = result
    
    # Try 4 blocks
    for combo in combinations(all_blocks, 4):
        if time.time() > deadline:
            break
        result = _try_block_config(list(combo))
        if result and result[0] < best_rank:
            best_rank = result[0]
            best_result = result
    
    if best_result is None:
        # Fallback: schoolbook
        U_sb = np.zeros((n * m, n * m * p))
        V_sb = np.zeros((m * p, n * m * p))
        W_sb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    _, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
