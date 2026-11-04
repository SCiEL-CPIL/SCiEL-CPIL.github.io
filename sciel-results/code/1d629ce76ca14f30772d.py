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

def _build_rank84():
    """Build rank-84 decomposition by splitting 4x4x6 into three 4x4x2 subproblems.
    
    Each 4x4x2 subproblem uses 4 rank-7 Strassen 2x2*2x2 products.
    Total rank = 3 * 4 * 7 = 84.
    """
    R = 84
    U = np.zeros((16, R), dtype=np.int64)
    V = np.zeros((24, R), dtype=np.int64)
    W = np.zeros((24, R), dtype=np.int64)
    
    col = 0
    # Three independent 4x4x2 subproblems: columns 0-1, 2-3, 4-5 of output
    for p_block in range(3):
        # Each 4x4x2 block: 4 Strassen 2x2*2x2 products (2x2 grid of 2x2 blocks)
        for row_block in range(2):  # 2x2 blocks in row dimension (4 = 2*2)
            for col_block in range(2):  # 2x2 blocks in col dimension (4 = 2*2)
                # 7 Strassen terms for this 2x2*2x2 product
                for s in range(7):
                    # U: input A block (2x2)
                    for a in range(4):
                        if _SA[s][a]:
                            ar, ac_ = a >> 1, a & 1
                            # Map to global U index: (row_block*2 + ar)*4 + (col_block*2 + ac_)
                            u = (row_block*2 + ar)*4 + (col_block*2 + ac_)
                            U[u, col] += _SA[s][a]
                    
                    # V: input B block (2x2) for this p_block
                    for b in range(4):
                        if _SB[s][b]:
                            br, bc_ = b >> 1, b & 1
                            # Map to global V index: (col_block*2 + br)*6 + (p_block*2 + bc_)
                            v = (col_block*2 + br)*6 + (p_block*2 + bc_)
                            V[v, col] += _SB[s][b]
                    
                    # W: output C block (2x2) for this p_block
                    for (a, b, ga) in _SR[s]:
                        # Map to global W index: (p_block*2 + b)*4 + (row_block*2 + a)
                        w = (p_block*2 + b)*4 + (row_block*2 + a)
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

def _try_reduce_rank(U, V, W, max_steps=500, tol=1e-9):
    """Try to reduce rank by zeroing columns and adjusting others."""
    U, V, W = U.copy(), V.copy(), W.copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute tensor entries
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
    
    # Try removing columns one by one
    U0, V0, W0 = U.copy(), V.copy(), W.copy()
    current_U, current_V, current_W = U0.copy(), V0.copy(), W0.copy()
    current_R = R
    removed = set()
    
    for trial in range(R):
        if time.time() > _T_LIMIT - 5:
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
    fU = np.round(fU).astype(np.int64)
    fV = np.round(fV).astype(np.int64)
    fW = np.round(fW).astype(np.int64)
    
    final_err = check_err(fU, fV, fW)
    if final_err < 1e-8:
        return fU, fV, fW, final_R
    else:
        return U0, V0, W0, R

def construct():
    t0 = time.time()
    U, V, W = _build_rank84()
    
    # Verify base decomposition
    err = _verify(U, V, W)
    
    # Try to reduce rank
    if time.time() < _T_LIMIT - 10:
        U2, V2, W2, R2 = _try_reduce_rank(U, V, W)
        if R2 < U.shape[1]:
            U, V, W = U2, V2, W2
    
    def _to_int_list(mat):
        return [[int(x) for x in row] for row in mat]
    
    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
