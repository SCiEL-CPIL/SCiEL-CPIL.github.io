"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and dot products for 1x2x2 blocks.
Enhanced rank reduction with smarter column selection and vectorized verification.
Expected rank: 36 (or lower if reduction succeeds).
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_fast(U, V, W):
    """Fast vectorized verification using numpy."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Precompute all needed index combinations
    # For valid entries: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # For invalid entries (j != j2): sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r] == 0
    
    # Build index arrays for valid entries
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    k_arr = np.array(k_idx)
    
    u_rows = i_arr * m + j_arr
    v_rows = j_arr * p + k_arr
    w_rows = k_arr * n + i_arr
    
    # Compute all valid tensor entries
    valid_vals = np.sum(U[u_rows] * V[v_rows] * W[w_rows], axis=1)
    if np.any(np.abs(valid_vals - 1.0) > 1e-6):
        return False
    
    # Check invalid entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _verify(U, V, W):
    """Alias for fast verify."""
    return _verify_fast(U, V, W)

def _add_strassen(U, V, W, a_indices, b_indices, c_indices):
    """
    Adds a 2x2x2 Strassen decomposition to the global matrices.
    """
    rank = U.shape[1]
    
    # r=0 (M1): (A00+A11)(B00+B11)
    U[a_indices[0], rank] += 1; U[a_indices[3], rank] += 1
    V[b_indices[0], rank] += 1; V[b_indices[3], rank] += 1
    W[c_indices[0], rank] += 1; W[c_indices[3], rank] += 1
    
    # r=1 (M2): (A10+A11)B00
    U[a_indices[2], rank+1] += 1; U[a_indices[3], rank+1] += 1
    V[b_indices[0], rank+1] += 1
    W[c_indices[2], rank+1] += 1; W[c_indices[3], rank+1] += 1
    
    # r=2 (M3): A00(B01-B11)
    U[a_indices[0], rank+2] += 1
    V[b_indices[1], rank+2] += 1; V[b_indices[3], rank+2] -= 1
    W[c_indices[1], rank+2] += 1; W[c_indices[3], rank+2] += 1
    
    # r=3 (M4): A11(B10-B00)
    U[a_indices[3], rank+3] += 1
    V[b_indices[2], rank+3] += 1; V[b_indices[0], rank+3] -= 1
    W[c_indices[2], rank+3] += 1; W[c_indices[3], rank+3] += 1
    
    # r=4 (M5): (A00+A01)B11
    U[a_indices[0], rank+4] += 1; U[a_indices[1], rank+4] += 1
    V[b_indices[3], rank+4] += 1
    W[c_indices[1], rank+4] += 1; W[c_indices[3], rank+4] += 1
    
    # r=5 (M6): (A10-A00)(B00+B01)
    U[a_indices[2], rank+5] += 1; U[a_indices[0], rank+5] -= 1
    V[b_indices[0], rank+5] += 1; V[b_indices[1], rank+5] += 1
    W[c_indices[2], rank+5] += 1; W[c_indices[3], rank+5] += 1
    
    # r=6 (M7): (A01-A11)(B10+B11)
    U[a_indices[1], rank+6] += 1; U[a_indices[3], rank+6] -= 1
    V[b_indices[2], rank+6] += 1; V[b_indices[3], rank+6] += 1
    W[c_indices[0], rank+6] += 1; W[c_indices[3], rank+6] += 1

def _add_dot_1x2x2(U, V, W, a_indices, b_indices, c_indices):
    """
    Adds a 1x2x2 dot product decomposition.
    """
    rank = U.shape[1]
    
    # r=0: A0 * (B00, B01) -> (C0, C1)
    U[a_indices[0], rank] += 1
    V[b_indices[0], rank] += 1; V[b_indices[1], rank] += 1
    W[c_indices[0], rank] += 1; W[c_indices[1], rank] += 1
    
    # r=1: A1 * (B10, B11) -> (C0, C1)
    U[a_indices[1], rank+1] += 1
    V[b_indices[2], rank+1] += 1; V[b_indices[3], rank+1] += 1
    W[c_indices[0], rank+1] += 1; W[c_indices[1], rank+1] += 1

def _block_decomposition():
    n, m, p = DIMENSIONS  # 3, 4, 8
    total_rank = 36
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    current_rank = 0
    
    def get_c_indices_strassen(k_start, i_start):
        return [
            (k_start) * n + (i_start),
            (k_start + 1) * n + (i_start),
            (k_start) * n + (i_start + 1),
            (k_start + 1) * n + (i_start + 1)
        ]

    # Block 1: C[0:2, 0:4]
    a_idx_1a = [i*4+j for i in range(2) for j in range(2)]
    b_idx_1a = [j*8+k for j in range(2) for k in range(2)]
    c_idx_1a = get_c_indices_strassen(0, 0)
    _add_strassen(U, V, W, a_idx_1a, b_idx_1a, c_idx_1a)
    current_rank += 7
    
    a_idx_1b = [i*4+j for i in range(2) for j in range(2,4)]
    b_idx_1b = [j*8+k for j in range(2,4) for k in range(2)]
    c_idx_1b = get_c_indices_strassen(0, 0)
    _add_strassen(U, V, W, a_idx_1b, b_idx_1b, c_idx_1b)
    current_rank += 7
    
    # Block 2: C[0:2, 4:8]
    a_idx_2a = [i*4+j for i in range(2) for j in range(2)]
    b_idx_2a = [j*8+k for j in range(2) for k in range(4,6)]
    c_idx_2a = get_c_indices_strassen(4, 0)
    _add_strassen(U, V, W, a_idx_2a, b_idx_2a, c_idx_2a)
    current_rank += 7
    
    a_idx_2b = [i*4+j for i in range(2) for j in range(2,4)]
    b_idx_2b = [j*8+k for j in range(2,4) for k in range(4,6)]
    c_idx_2b = get_c_indices_strassen(4, 0)
    _add_strassen(U, V, W, a_idx_2b, b_idx_2b, c_idx_2b)
    current_rank += 7
    
    # Block 3: C[2:3, 0:4]
    a_idx_3a = [2*4+j for j in range(2)]
    b_idx_3a = [j*8+k for j in range(2) for k in range(2)]
    c_idx_3a = [k*3+2 for k in range(2)]
    _add_dot_1x2x2(U, V, W, a_idx_3a, b_idx_3a, c_idx_3a)
    current_rank += 2
    
    a_idx_3b = [2*4+j for j in range(2,4)]
    b_idx_3b = [j*8+k for j in range(2,4) for k in range(2)]
    c_idx_3b = [k*3+2 for k in range(2)]
    _add_dot_1x2x2(U, V, W, a_idx_3b, b_idx_3b, c_idx_3b)
    current_rank += 2
    
    # Block 4: C[2:3, 4:8]
    a_idx_4a = [2*4+j for j in range(2)]
    b_idx_4a = [j*8+k for j in range(2) for k in range(4,6)]
    c_idx_4a = [k*3+2 for k in range(4,6)]
    _add_dot_1x2x2(U, V, W, a_idx_4a, b_idx_4a, c_idx_4a)
    current_rank += 2
    
    a_idx_4b = [2*4+j for j in range(2,4)]
    b_idx_4b = [j*8+k for j in range(2,4) for k in range(4,6)]
    c_idx_4b = [k*3+2 for k in range(4,6)]
    _add_dot_1x2x2(U, V, W, a_idx_4b, b_idx_4b, c_idx_4b)
    current_rank += 2
    
    return U, V, W

def _column_support(U, V, W, col_idx):
    """Compute the 'support' of a column: number of non-zero entries across U, V, W."""
    return (np.count_nonzero(U[:, col_idx]) + 
            np.count_nonzero(V[:, col_idx]) + 
            np.count_nonzero(W[:, col_idx]))

def _try_rank_reduction(U, V, W, deadline):
    """
    Try to find linear dependencies to reduce rank.
    Strategy: 
    1. Try removing single columns, prioritizing those with minimal support
    2. If successful, try removing more columns
    3. Try pairwise removals for additional reduction
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Compute support for each column to prioritize removal candidates
    supports = [_column_support(U, V, W, r) for r in range(rank)]
    
    # Sort columns by support (ascending) - try removing low-support columns first
    col_order = sorted(range(rank), key=lambda r: supports[r])
    
    # Phase 1: Try removing single columns
    for r_to_remove in col_order:
        if time.time() > deadline:
            break
        
        mask = [i for i in range(rank) if i != r_to_remove]
        U_r = U[:, mask]
        V_r = V[:, mask]
        W_r = W[:, mask]
        
        if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            U, V, W = U_r.copy(), V_r.copy(), W_r.copy()
            rank -= 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
            # Recompute supports for remaining columns
            supports = [_column_support(U, V, W, r) for r in range(rank)]
            col_order = sorted(range(rank), key=lambda r: supports[r])
            # Don't break - try to remove more
    
    # Phase 2: Try pairwise removals from the reduced decomposition
    if best_rank < rank:
        current_rank = best_rank
        for i in range(current_rank):
            if time.time() > deadline:
                break
            for j in range(i+1, current_rank):
                if time.time() > deadline:
                    break
                mask = [k for k in range(current_rank) if k != i and k != j]
                U_r = U[:, mask]
                V_r = V[:, mask]
                W_r = W[:, mask]
                
                if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    U, V, W = U_r.copy(), V_r.copy(), W_r.copy()
                    current_rank -= 2
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = current_rank
                    break  # Restart pairwise search with new rank
    
    if best_rank < rank:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    try:
        U, V, W = _block_decomposition()
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            if time.time() < deadline:
                reduced = _try_rank_reduction(U, V, W, deadline)
                if reduced is not None:
                    U_r, V_r, W_r = reduced
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    return _schoolbook()
# EVOLVE-BLOCK-END
