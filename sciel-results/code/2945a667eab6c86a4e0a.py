"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# Precompute index structures for fast ALS
def _precompute_indices():
    n, m, p = DIMENSIONS
    U_indices = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = [(j * p + k, k * n + i) for k in range(p)]
            U_indices.append(pairs)
    
    V_indices = []
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = [(i * m + j, kk * n + i) for i in range(n) for kk in range(p)]
            V_indices.append(pairs)
    
    W_indices = []
    for k in range(p):
        for i in range(n):
            c = k * n + i
            pairs = [(ii * m + j, j * p + k) for ii in range(n) for j in range(m)]
            W_indices.append(pairs)
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    return U_indices, V_indices, W_indices, target

_U_IDX, _V_IDX, _W_IDX, _TARGET = _precompute_indices()

def _compute_tensor(U_arr, V_arr, W_arr):
    """Compute the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    n, m, p = DIMENSIONS
    rank = U_arr.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    return T

def _max_error(T):
    """Compute max error on target and non-target entries."""
    n, m, p = DIMENSIONS
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    non_target_err = np.max(np.abs(T[~_TARGET])) if (~_TARGET).any() else 0.0
    return max_err, non_target_err

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T = _compute_tensor(U_arr, V_arr, W_arr)
    max_err, non_target_err = _max_error(T)
    return max_err <= tol and non_target_err <= tol

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
    Uses Strassen rank-7 for each 2x2x2 sub-block + 4 rank-1 for the third column.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
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

def _als_decompose(rank, max_iters=30, deadline=None, seed=None):
    """
    Vectorized ALS for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    U = rng.randn(n*m, rank) * 0.3
    V = rng.randn(m*p, rank) * 0.3
    W = rng.randn(p*n, rank) * 0.3
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U rows
        for a in range(n*m):
            pairs = _U_IDX[a]
            M = np.array([V[b, :] * W[c, :] for b, c in pairs])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                U[a, :] = x
            except:
                pass
        
        # Update V rows
        for b in range(m*p):
            pairs = _V_IDX[b]
            M = np.array([U[a, :] * W[c, :] for a, c in pairs])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                V[b, :] = x
            except:
                pass
        
        # Update W rows
        for c in range(p*n):
            pairs = _W_IDX[c]
            M = np.array([U[a, :] * V[b, :] for a, b in pairs])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                W[c, :] = x
            except:
                pass
        
        T = _compute_tensor(U, V, W)
        max_err, non_target_err = _max_error(T)
        cur_err = max(max_err, non_target_err)
        
        if cur_err < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
        
        if cur_err > prev_err * 2.0:
            break
        prev_err = cur_err
    
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
    deadline = start_time + 140  # Safety margin (150s limit)
    
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
    
    # Try to find lower rank with ALS
    if best_rank > 40 and time.time() < deadline:
        for rank in range(36, best_rank):
            if time.time() > deadline - 5:
                break
            
            found = False
            for seed in range(8):
                if time.time() > deadline - 5:
                    break
                
                result = _als_decompose(rank, max_iters=25, deadline=deadline, seed=seed)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                        found = True
                        break
            
            if found:
                continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
