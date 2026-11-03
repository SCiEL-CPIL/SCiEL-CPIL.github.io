"""
Matrix multiplication 2x4x7 - Rank-minimal exact tensor decomposition.

Uses Strassen (rank 7) for 2x2x2 blocks and schoolbook (rank 4) for 2x2x1 blocks.
Total rank: 2*3*7 + 2*1*4 = 42 + 8 = 50.

Strategy:
1. Build the known rank-50 decomposition.
2. Try to find a lower-rank decomposition via structured search.
3. Return the best valid decomposition found.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _build_rank50():
    """Build the known rank-50 decomposition using Strassen and schoolbook blocks."""
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            p_sub = k_end - k_start
            m_sub = j_end - j_start
            
            for i in range(n):
                for j_sub in range(m_sub):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            for j_sub in range(m_sub):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            col_offset += r_sub
    
    return U, V, W

def _try_merge_columns(U, V, W, n, m, p, max_time):
    """
    Try to reduce rank by merging columns. 
    For each pair of columns, try to find a linear combination that preserves the identity.
    This is a heuristic approach.
    """
    start_time = time.time()
    r = U.shape[1]
    
    # Try to find linear dependencies among columns
    # If columns c1 and c2 are linearly dependent, we can merge them
    # This is a simplified approach
    
    # For each pair, check if they can be merged
    for c1 in range(r):
        if time.time() - start_time > max_time:
            break
        for c2 in range(c1 + 1, r):
            if time.time() - start_time > max_time:
                break
            
            # Check if columns c1 and c2 are linearly dependent
            # This is a heuristic: try to find alpha, beta such that
            # alpha * U[:, c1] + beta * U[:, c2] = 0 (or similar)
            
            # For simplicity, just check if the columns are identical or negatives
            if np.allclose(U[:, c1], U[:, c2]) and np.allclose(V[:, c1], V[:, c2]) and np.allclose(W[:, c1], W[:, c2]):
                # Columns are identical, can merge
                new_U = np.delete(U, c2, axis=1)
                new_V = np.delete(V, c2, axis=1)
                new_W = np.delete(W, c2, axis=1)
                if _verify(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W
            
            elif np.allclose(U[:, c1], -U[:, c2]) and np.allclose(V[:, c1], -V[:, c2]) and np.allclose(W[:, c1], -W[:, c2]):
                # Columns are negatives, can merge
                new_U = np.delete(U, c2, axis=1)
                new_V = np.delete(V, c2, axis=1)
                new_W = np.delete(W, c2, axis=1)
                if _verify(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W
    
    return None

def _try_local_search(U, V, W, n, m, p, target_rank, max_time):
    """
    Try to reduce rank via local search.
    Start from the given decomposition and try to set some entries to zero.
    """
    start_time = time.time()
    r = U.shape[1]
    
    if r <= target_rank:
        return U, V, W
    
    # Try to zero out entries and see if the identity still holds
    # This is a heuristic approach
    
    # For each column, try to see if it can be removed
    for c in range(r):
        if time.time() - start_time > max_time:
            break
        if U.shape[1] <= target_rank:
            break
        
        # Try removing column c
        new_U = np.delete(U, c, axis=1)
        new_V = np.delete(V, c, axis=1)
        new_W = np.delete(W, c, axis=1)
        
        if _verify(new_U, new_V, new_W, n, m, p):
            U, V, W = new_U, new_V, new_W
            r = U.shape[1]
    
    if U.shape[1] <= target_rank:
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build the known rank-50 decomposition
    U, V, W = _build_rank50()
    
    # Verify the rank-50 decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        r = n * m * p
        U_fb = np.zeros((n * m, r))
        V_fb = np.zeros((m * p, r))
        W_fb = np.zeros((p * n, r))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce rank via merging columns
    if time.time() < deadline - 30:
        merged = _try_merge_columns(U, V, W, n, m, p, max_time=20)
        if merged is not None:
            U_m, V_m, W_m = merged
            if U_m.shape[1] < best_rank:
                best_U, best_V, best_W = U_m, V_m, W_m
                best_rank = U_m.shape[1]
    
    # Try local search to reduce rank further
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
            if time.time() > deadline - 5:
                break
            result = _try_local_search(best_U, best_V, best_W, n, m, p, target_rank, max_time=5)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p):
                    if U_r.shape[1] < best_rank:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = U_r.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
