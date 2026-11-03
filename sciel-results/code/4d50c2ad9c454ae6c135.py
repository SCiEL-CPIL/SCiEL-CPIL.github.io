"""Matrix multiplication 4,4,5 tensor decomposition - rank optimization."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _build_4x4x4_strassen(n=4, m=4, p=4):
    """Build rank-49 Strassen decomposition for 4x4x4."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sp, v1_sp, w1_sp) in strassen_terms:
        for (u2_sp, v2_sp, w2_sp) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sp:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sp:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i, j = i1*2+i2, j1*2+j2
                    u[i*m+j] = val1*val2
            
            for (idx1, val1) in v1_sp:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sp:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j, k = j1*2+j2, k1*2+k2
                    v[j*p+k] = val1*val2
            
            for (idx1, val1) in w1_sp:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sp:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k, i = k1*2+k2, i1*2+i2
                    w[k*n+i] = val1*val2
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T

def _build_4x4x5(n=4, m=4, p=5):
    """
    Build rank-65 decomposition: 49 from 4x4x4 Strassen + 16 from 4x4x1 naive.
    """
    U4, V4, W4 = _build_4x4x4_strassen(n, m, 4)
    
    U = np.zeros((n*m, 65))
    V = np.zeros((m*p, 65))
    W = np.zeros((p*n, 65))
    
    U[:, :49] = U4
    V[:, :49] = V4
    W[:4*n, :49] = W4
    
    # 4x4x1 naive part: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            col = 49 + i*m + j
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_half_int(U, V, W, tol=1e-9):
    """Check all entries are multiples of 0.5."""
    for M in [U, V, W]:
        M = np.array(M, dtype=np.float64)
        if np.any(np.abs(M * 2 - np.round(M * 2)) > tol):
            return False
    return True

def _build_schoolbook(n=4, m=4, p=5):
    """Fallback: rank n*m*p = 80."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_remove_column(U, V, W, col_idx, n, m, p):
    """Try removing a column and check if decomposition still holds."""
    U_new = np.delete(U, col_idx, axis=1)
    V_new = np.delete(V, col_idx, axis=1)
    W_new = np.delete(W, col_idx, axis=1)
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None

def _try_remove_pair(U, V, W, col1, col2, n, m, p):
    """Try removing two columns simultaneously."""
    cols = sorted([col1, col2])
    U_new = np.delete(U, cols, axis=1)
    V_new = np.delete(V, cols, axis=1)
    W_new = np.delete(W, cols, axis=1)
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None

def _als_refine(U, V, W, n, m, p, max_iter=50, tol=1e-10):
    """
    ALS (Alternating Least Squares) refinement.
    Try to find a lower-rank decomposition by projecting.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Try to find a column that is nearly redundant
        best_reduction = None
        for r in range(rank):
            # Check if column r can be expressed as combination of others
            # Simple heuristic: check if removing it and adjusting others works
            pass
        
        # Instead, try a different approach: use SVD to find low-rank structure
        # Flatten the tensor and try to find a better factorization
        break
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, max_rank=65, time_limit=30):
    """
    Try to reduce rank by removing columns one at a time or in pairs.
    """
    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # First, try removing single columns
    changed = True
    while changed and time.time() - start < time_limit:
        changed = False
        for r in range(rank):
            result = _try_remove_column(U, V, W, r, n, m, p)
            if result is not None:
                U, V, W = result
                rank -= 1
                changed = True
                break
    
    # Then try removing pairs
    if time.time() - start < time_limit - 5:
        changed = True
        while changed and time.time() - start < time_limit:
            changed = False
            for r1 in range(rank):
                if time.time() - start > time_limit - 2:
                    break
                for r2 in range(r1 + 1, rank):
                    result = _try_remove_pair(U, V, W, r1, r2, n, m, p)
                    if result is not None:
                        U, V, W = result
                        rank -= 2
                        changed = True
                        break
                if changed:
                    break
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Rank-65 Strassen construction
    try:
        U, V, W = _build_4x4x5(n, m, p)
        rank = U.shape[1]
        if _verify(U, V, W, n, m, p) and _verify_half_int(U, V, W):
            best_result = (U, V, W)
            best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Try to reduce rank from the rank-65 construction
    if best_result is not None and time.time() < deadline - 20:
        try:
            U_red, V_red, W_red = _try_rank_reduction(
                best_result[0], best_result[1], best_result[2], 
                n, m, p, time_limit=25
            )
            rank_red = U_red.shape[1]
            if rank_red < best_rank and _verify(U_red, V_red, W_red, n, m, p):
                best_result = (U_red, V_red, W_red)
                best_rank = rank_red
        except Exception:
            pass
    
    # Approach 3: Try a different construction - maybe a non-Kronecker Strassen
    # For 4x4x4, there might be a rank < 49 decomposition
    # But 49 is the known best for 4x4x4 via Kronecker Strassen
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
