import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_N, _M, _P = 2, 3, 8
_NMP = _N * _M * _P  # 48

# Precompute target indices and mask
_TARGET_IDX = []
for _i in range(_N):
    for _j in range(_M):
        for _k in range(_P):
            _TARGET_IDX.append((_i * _M + _j, _j * _P + _k, _k * _N + _i))

_TARGET_MASK = np.zeros((_N*_M, _M*_P, _P*_N), dtype=bool)
for _a, _b, _c in _TARGET_IDX:
    _TARGET_MASK[_a, _b, _c] = True

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = 0.0
    for a, b, c in _TARGET_IDX:
        err = abs(T[a,b,c] - 1.0)
        if err > max_err:
            max_err = err
        if max_err > tol:
            return False
    
    non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
    return non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _als_decompose(rank, max_iters=100, deadline=None, seed=None, init=None):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    if init is not None:
        U = np.array(init[0], dtype=float).copy()
        V = np.array(init[1], dtype=float).copy()
        W = np.array(init[2], dtype=float).copy()
    else:
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    best_residual = np.inf
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            i = a_idx // m
            j = a_idx % m
            M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            j = b_idx // p
            k = b_idx % p
            rows = []
            for i in range(n):
                for kk in range(p):
                    a = i * m + j
                    c = kk * n + i
                    rows.append(U[a, :] * W[c, :])
            M = np.array(rows)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)
                V[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            k = c_idx // n
            i = c_idx % n
            rows = []
            for ii in range(n):
                for j in range(m):
                    a = ii * m + j
                    b = j * p + k
                    rows.append(U[a, :] * V[b, :])
            M = np.array(rows)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)
                W[c_idx, :] = x
            except:
                pass
                    
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
            
        max_err = 0.0
        for (a, b, c) in _TARGET_IDX:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        
        non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # First, try the explicit rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception:
        pass
    
    # Try to find rank 43 or lower
    if best_rank == 44 and time.time() < deadline - 10:
        # Try ALS for rank 43 using rank 44 as init
        # We need to project rank 44 to rank 43.
        # Simple projection: drop one column? No, that breaks it.
        # We can use the rank 44 solution to initialize ALS at rank 43 by taking a random 43-dim subspace.
        # Or just run ALS with random init for rank 43.
        
        for rank in [43, 42, 41, 40]:
            if time.time() > deadline - 5:
                break
            
            success = False
            # Try a few random inits
            for trial in range(5):
                if time.time() > deadline - 5:
                    break
                result = _als_decompose(rank, max_iters=50, deadline=deadline, seed=trial*1000+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        success = True
                        break
            if success:
                break
                
    # Final polish
    if best_rank < schoolbook_rank and time.time() < deadline - 2:
        # No specific polish implemented in this version to save time, 
        # but we can try rounding again
        rounded = _round_and_verify(best_U, best_V, best_W)
        if rounded:
            best_U, best_V, best_W = rounded
    
    return {"U": best_U, "V": best_V, "W": best_W}
