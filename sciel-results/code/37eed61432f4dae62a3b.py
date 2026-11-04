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
    U = np.zeros((16, R), dtype=np.int64)
    V = np.zeros((24, R), dtype=np.int64)
    W = np.zeros((24, R), dtype=np.int64)
    
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

def _smooth_reduce(U, V, W, max_iter=200, tol=1e-8):
    """Use smooth optimization to try to zero out columns while preserving decomposition."""
    n, m, p = _DIMS
    R = U.shape[1]
    
    # Precompute tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_entries.append((i*m+j, j*p+k, k*n+i))
    
    def compute_loss_and_grads(Uf, Vf, Wf, lambda_reg=1e-4):
        """Compute smooth loss: sum of squared errors + regularization to push columns toward zero."""
        loss = 0.0
        grad_U = np.zeros_like(Uf)
        grad_V = np.zeros_like(Vf)
        grad_W = np.zeros_like(Wf)
        
        for (ui, vi, wi) in tensor_entries:
            # val = sum_r U[ui,r] * V[vi,r] * W[wi,r]
            # d_loss/dU[ui,r] = 2*(val-1) * V[vi,r] * W[wi,r]
            val = np.dot(Uf[ui,:] * Vf[vi,:], Wf[wi,:])
            err = val - 1.0
            factor = 2.0 * err
            for r in range(R):
                uv = Vf[vi, r] * Wf[wi, r]
                vw = Uf[ui, r] * Wf[wi, r]
                wv = Uf[ui, r] * Vf[vi, r]
                grad_U[ui, r] += factor * uv
                grad_V[vi, r] += factor * vw
                grad_W[wi, r] += factor * wv
            loss += err * err
        
        # Regularization: encourage columns to be zero (L2 penalty on each column norm)
        for r in range(R):
            col_U = Uf[:, r]
            col_V = Vf[:, r]
            col_W = Wf[:, r]
            # Gradient of lambda * (||col_U||^2 + ||col_V||^2 + ||col_W||^2)
            grad_U[:, r] += 2 * lambda_reg * col_U
            grad_V[:, r] += 2 * lambda_reg * col_V
            grad_W[:, r] += 2 * lambda_reg * col_W
            loss += lambda_reg * (np.dot(col_U, col_U) + np.dot(col_V, col_V) + np.dot(col_W, col_W))
        
        return loss, grad_U, grad_V, grad_W
    
    # Start from current solution
    Uf = U.astype(np.float64).copy()
    Vf = V.astype(np.float64).copy()
    Wf = W.astype(np.float64).copy()
    
    lr = 0.01
    best_loss = float('inf')
    best_Uf, best_Vf, best_Wf = Uf.copy(), Vf.copy(), Wf.copy()
    
    t0 = time.time()
    for it in range(max_iter):
        if time.time() > _T_LIMIT - 10:
            break
        
        loss, gU, gV, gW = compute_loss_and_grads(Uf, Vf, Wf)
        
        if loss < best_loss:
            best_loss = loss
            best_Uf, best_Vf, best_Wf = Uf.copy(), Vf.copy(), Wf.copy()
        
        # Gradient descent step
        Uf -= lr * gU
        Vf -= lr * gV
        Wf -= lr * gW
        
        # Decay learning rate
        lr *= 0.995
        
        # Check if we can round and still be valid
        if it % 20 == 0:
            Ur = np.round(Uf).astype(np.int64)
            Vr = np.round(Vf).astype(np.int64)
            Wr = np.round(Wf).astype(np.int64)
            err = check_err(Ur, Vr, Wr)
            if err < 1e-6:
                # Count non-zero columns
                nz_U = np.sum(np.any(Ur != 0, axis=0))
                nz_V = np.sum(np.any(Vr != 0, axis=0))
                nz_W = np.sum(np.any(Wr != 0, axis=0))
                effective_rank = min(nz_U, nz_V, nz_W)
                if effective_rank < R:
                    return Ur, Vr, Wr, effective_rank
    
    # Try rounding the best found
    Ur = np.round(best_Uf).astype(np.int64)
    Vr = np.round(best_Vf).astype(np.int64)
    Wr = np.round(best_Wf).astype(np.int64)
    err = check_err(Ur, Vr, Wr)
    if err < 1e-6:
        nz_U = np.sum(np.any(Ur != 0, axis=0))
        nz_V = np.sum(np.any(Vr != 0, axis=0))
        nz_W = np.sum(np.any(Wr != 0, axis=0))
        effective_rank = min(nz_U, nz_V, nz_W)
        if effective_rank < R:
            return Ur, Vr, Wr, effective_rank
    
    return U, V, W, R

def _try_single_removal(U, V, W, tol=1e-8):
    """Try removing each single column and check if decomposition still holds."""
    n, m, p = _DIMS
    R = U.shape[1]
    
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
    
    # Try each single column removal
    for c in range(R):
        if time.time() > _T_LIMIT - 5:
            break
        
        U2 = U.copy()
        V2 = V.copy()
        W2 = W.copy()
        U2[:, c] = 0
        V2[:, c] = 0
        W2[:, c] = 0
        
        err = check_err(U2, V2, W2)
        if err < tol:
            # Remove this column
            keep = [i for i in range(R) if i != c]
            return U2[:, keep], V2[:, keep], W2[:, keep], R - 1
    
    return U, V, W, R

def construct():
    t0 = time.time()
    U, V, W = _build_rank77()
    
    # Verify base decomposition
    err = _verify(U, V, W)
    
    # Try single column removal first (fast)
    if time.time() < _T_LIMIT - 20:
        U2, V2, W2, R2 = _try_single_removal(U, V, W)
        if R2 < U.shape[1]:
            U, V, W = U2, V2, W2
    
    # Try greedy reduction
    if time.time() < _T_LIMIT - 15:
        U3, V3, W3, R3 = _try_reduce_rank(U, V, W)
        if R3 < U.shape[1]:
            U, V, W = U3, V3, W3
    
    # Try smooth optimization to find more reductions
    if time.time() < _T_LIMIT - 10:
        U4, V4, W4, R4 = _smooth_reduce(U, V, W)
        if R4 < U.shape[1]:
            U, V, W = U4, V4, W4
    
    def _to_int_list(mat):
        return [[int(x) for x in row] for row in mat]
    
    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
