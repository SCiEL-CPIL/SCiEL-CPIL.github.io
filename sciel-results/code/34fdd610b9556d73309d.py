import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    if r == 0:
        return False
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

def _try_different_blockings():
    """
    Try different blocking strategies to find the minimum rank.
    """
    best_rank = float('inf')
    best_result = None
    
    # Strategy 1: Two 2x2x2 Strassen blocks on (i,j) in {0,1}^2, k in {0,1} and {2,3}
    # Plus handle i=2 row
    U_cols = []
    V_cols = []
    W_cols = []
    
    for kb in [0, 2]:
        U_terms, V_terms, W_terms = _strassen_2x2_block(0, 0, kb)
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
    
    # Handle i=2: terms (2, j, k) for j=0,1,2 and k=0,1,2,3
    # Group by j and pair k's
    for j in range(m):
        ks = [0, 1, 2, 3]
        # Pair (k=0,k=1) and (k=2,k=3)
        for idx in range(0, 4, 2):
            k1 = ks[idx]
            k2 = ks[idx+1]
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            u_vec[2 * m + j] = 1
            v_vec[j * p + k1] = 1
            v_vec[j * p + k2] = 1
            w_vec[k1 * n + 2] = 1
            w_vec[k2 * n + 2] = 1
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]
    if rank < best_rank and _verify(U, V, W):
        best_rank = rank
        best_result = (U, V, W)
    
    # Strategy 2: One 2x2x2 Strassen on (i,j) in {0,1}^2, k in {0,1}
    # One 2x2x2 Strassen on (i,j) in {0,1}^2, k in {2,3}  
    # Cross-pair i=2 terms across j
    U_cols = []
    V_cols = []
    W_cols = []
    
    for kb in [0, 2]:
        U_terms, V_terms, W_terms = _strassen_2x2_block(0, 0, kb)
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
    
    # For i=2, cross-pair: (2,0,k) with (2,1,k) for each k, and (2,2,k) separately
    # Or pair across k: (2,j,k) with (2,j,k+2)
    for j in range(m):
        for k in [0, 1]:
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            u_vec[2 * m + j] = 1
            v_vec[j * p + k] = 1
            v_vec[j * p + k + 2] = 1
            w_vec[k * n + 2] = 1
            w_vec[(k+2) * n + 2] = 1
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]
    if rank < best_rank and _verify(U, V, W):
        best_rank = rank
        best_result = (U, V, W)
    
    # Strategy 3: Try 2x2x2 Strassen on (i,j) in {0,1}^2 for all k using two blocks,
    # but also try (i,j) in {1,2}^2
    for i_base in [0, 1]:
        U_cols = []
        V_cols = []
        W_cols = []
        
        for kb in [0, 2]:
            U_terms, V_terms, W_terms = _strassen_2x2_block(i_base, 0, kb)
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
        
        # Handle remaining i
        remaining_i = [i for i in range(n) if i not in [i_base, i_base+1]]
        for i in remaining_i:
            for j in range(m):
                ks = [0, 1, 2, 3]
                for idx in range(0, 4, 2):
                    k1 = ks[idx]
                    k2 = ks[idx+1]
                    u_vec = np.zeros(n * m)
                    v_vec = np.zeros(m * p)
                    w_vec = np.zeros(p * n)
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k1] = 1
                    v_vec[j * p + k2] = 1
                    w_vec[k1 * n + i] = 1
                    w_vec[k2 * n + i] = 1
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        rank = U.shape[1]
        if rank < best_rank and _verify(U, V, W):
            best_rank = rank
            best_result = (U, V, W)
    
    if best_result is not None:
        return best_result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try different blocking strategies
    result = _try_different_blockings()
    
    if result is not None:
        U, V, W = result
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to original approach
    strassen_covered = set()
    blocks = [
        (0, 0, 0), 
        (0, 0, 2)  
    ]
    
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
            
    remaining_terms = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    if (i, j) not in remaining_terms:
                        remaining_terms[(i, j)] = []
                    remaining_terms[(i, j)].append(k)
                    
    for (i, j), ks in remaining_terms.items():
        ks = sorted(ks)
        num_terms = len(ks)
        for idx in range(0, num_terms, 2):
            if idx + 1 < num_terms:
                k1 = ks[idx]
                k2 = ks[idx+1]
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_idx = i * m + j
                u_vec[u_idx] = 1
                
                v_idx1 = j * p + k1
                v_idx2 = j * p + k2
                v_vec[v_idx1] = 1
                v_vec[v_idx2] = 1
                
                w_idx1 = k1 * n + i
                w_idx2 = k2 * n + i
                w_vec[w_idx1] = 1
                w_vec[w_idx2] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
            else:
                k1 = ks[idx]
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_vec[i * m + j] = 1
                v_vec[j * p + k1] = 1
                w_vec[k1 * n + i] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
                
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
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
    
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
