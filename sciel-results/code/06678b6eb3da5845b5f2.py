import numpy as np
import time

_DIMS = (4, 4, 6)
_T_LIMIT = 120.0

# Strassen 2x2x2 coefficients
_SA = [(1,0,0,1),(0,0,1,1),(1,0,0,0),(0,0,0,1),(1,1,0,0),(-1,0,1,0),(0,1,0,-1)]
_SB = [(1,0,0,1),(1,0,0,0),(0,1,0,-1),(-1,0,1,0),(0,0,0,1),(1,1,0,0),(0,0,1,1)]
_SR = [
    ((0,0,1),(1,1,1)),
    ((1,0,1),(1,1,-1)),
    ((0,1,1),(1,1,1)),
    ((0,0,1),(1,0,1)),
    ((0,0,-1),(0,1,1)),
    ((1,1,1),),
    ((0,0,1),)
]

def _build_rank77():
    """Build the rank-77 Strassen tiling for 4x4 * 4x6."""
    R = 77
    U = np.zeros((16, R), dtype=np.float64)
    V = np.zeros((24, R), dtype=np.float64)
    W = np.zeros((24, R), dtype=np.float64)
    
    col = 0
    for s in range(7):
        # 7 inner Strassen 2x2x2 terms
        for si in range(7):
            for a in range(4):
                if _SA[si][a]:
                    ar, ac_ = a >> 1, a & 1
                    for r in range(4):
                        if _SA[s][r]:
                            rr, rc = r >> 1, r & 1
                            u = (rr*2 + ar)*4 + (rc*2 + ac_)
                            U[u, col] += _SA[si][a] * _SA[s][r]
            for b in range(4):
                if _SB[si][b]:
                    br, bc_ = b >> 1, b & 1
                    for c in range(4):
                        if _SB[s][c]:
                            cr, cc = c >> 1, c & 1
                            v = (cr*2 + br)*6 + (cc*3 + bc_)
                            V[v, col] += _SB[si][b] * _SB[s][c]
            for (a, b, ga) in _SR[s]:
                for (za, zb, gz) in _SR[si]:
                    w = (b*3 + zb)*4 + (a*2 + za)
                    W[w, col] += ga * gz
            col += 1
        
        # 4 schoolbook 2x2x1 terms (for the 3rd block column)
        for u2 in range(2):
            for v2 in range(2):
                for r in range(4):
                    if _SA[s][r]:
                        rr, rc = r >> 1, r & 1
                        u = (rr*2 + u2)*4 + (rc*2 + v2)
                        U[u, col] += _SA[s][r]
                for c in range(4):
                    if _SB[s][c]:
                        cr, cc = c >> 1, c & 1
                        v = (cr*2 + v2)*6 + (cc*3 + 2)
                        V[v, col] += _SB[s][c]
                for (a, b, ga) in _SR[s]:
                    w = (b*3 + 2)*4 + (a*2 + u2)
                    W[w, col] += ga
                col += 1
    
    return U, V, W

def _verify(U, V, W, tol=1e-10):
    """Check if decomposition is exact. Returns max error."""
    n, m, p = _DIMS
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_reduce_rank(U, V, W, max_steps=2000, tol=1e-9):
    """Try to reduce rank by zeroing columns and adjusting others."""
    U, V, W = U.copy(), V.copy(), W.copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute tensor entries for quick checking
    n, m, p = _DIMS
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_entries.append((i*m+j, j*p+k, k*n+i))
    
    def check_err(matU, matV, matW):
        mx = 0.0
        for (ui, vi, wi) in tensor_entries:
            val = np.dot(matU[ui,:] * matV[vi,:], matW[wi,:])
            e = abs(val - 1.0)
            if e > mx:
                mx = e
        return mx
    
    # Try zeroing each column and solving least-squares correction
    for trial in range(R):
        if time.time() > _T_LIMIT - 10:
            break
        col_idx = trial
        # Zero out this column
        U[:, col_idx] = 0
        V[:, col_idx] = 0
        W[:, col_idx] = 0
        
        # Try small perturbations of remaining columns to fix
        # This is a simplified approach - just check if error is small
        err = check_err(U, V, W)
        if err < tol:
            best_rank = best_rank - 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        else:
            # Restore
            U[:, col_idx] = 0  # already zero
            # We need the original values back - this approach is flawed
            # Let's use a different strategy
            break
    
    # Better approach: try removing columns one at a time from a fresh copy
    U0, V0, W0 = _build_rank77()
    current_U, current_V, current_W = U0.copy(), V0.copy(), W0.copy()
    current_R = R
    removed = set()
    
    for trial in range(R):
        if time.time() > _T_LIMIT - 8:
            break
        # Find the column with smallest norm
        min_norm = float('inf')
        min_col = -1
        for c in range(current_R):
            if c in removed:
                continue
            norm = (np.sum(current_U[:,c]**2) + np.sum(current_V[:,c]**2) + np.sum(current_W[:,c]**2)) ** 0.5
            if norm < min_norm:
                min_norm = norm
                min_col = c
        if min_col < 0:
            break
        
        # Try zeroing this column
        current_U[:, min_col] = 0
        current_V[:, min_col] = 0
        current_W[:, min_col] = 0
        
        err = check_err(current_U, current_V, current_W)
        if err < tol:
            removed.add(min_col)
        else:
            current_U[:, min_col] = U0[:, min_col]
            current_V[:, min_col] = V0[:, min_col]
            current_W[:, min_col] = W0[:, min_col]
            break
    
    # Compact the remaining columns
    keep_cols = [c for c in range(R) if c not in removed]
    final_R = len(keep_cols)
    fU = current_U[:, keep_cols]
    fV = current_V[:, keep_cols]
    fW = current_W[:, keep_cols]
    
    # Round to integers
    fU = np.round(fU)
    fV = np.round(fV)
    fW = np.round(fW)
    
    final_err = check_err(fU, fV, fW)
    if final_err < 1e-8:
        return fU, fV, fW, final_R
    else:
        return U0, V0, W0, R

def construct():
    t0 = time.time()
    U, V, W = _build_rank77()
    
    # Verify base decomposition
    err = _verify(U, V, W)
    
    # Try to reduce rank
    if time.time() < _T_LIMIT - 15:
        U2, V2, W2, R2 = _try_reduce_rank(U, V, W)
        if R2 < U.shape[1]:
            U, V, W = U2, V2, W2
    
    def _to_int_list(mat):
        return [[int(round(x)) for x in row] for row in mat]
    
    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
