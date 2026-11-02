"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _build_target():
    """Build the target multiplication tensor once."""
    n, m, p = DIMENSIONS
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return target

_TARGET = None
def _get_target():
    global _TARGET
    if _TARGET is None:
        _TARGET = _build_target()
    return _TARGET

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    target = _get_target()
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    err = np.max(np.abs(T - target))
    return err <= tol

def _max_err(U, V, W):
    """Return max error."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    target = _get_target()
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.max(np.abs(T - target))

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
    Split into 4 blocks of 2x3 * 3x2, each using rank-10 (Strassen-7 + 3 outer).
    """
    n, m, p = 2, 3, 8
    rank_per_block = 10
    total_rank = 4 * rank_per_block  # 40... wait, 4*10=40 not 44
    
    # Actually let me reconsider. For 2x3 * 3x2:
    # C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k]  for k=0,1
    # C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k]  for k=0,1
    # 
    # The 2x2 * 2x2 part (using cols 0,1 of A and rows 0,1 of B) has rank 7 (Strassen)
    # The remaining column (col 2 of A) needs 2*2=4 outer products
    # Total per block: 7 + 4 = 11
    # 4 blocks: 44
    #
    # Wait, but we can do better. Let me use rank 10 per block if possible.
    # Actually 7+4=11 per block, 4*11=44. That matches.
    
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for the 2x2 * 2x2 sub-problem (A rows 0,1 x B cols k0,k1)
        # Indices: A[0]=a00, A[1]=a01, A[3]=a10, A[4]=a11 (row-major: row0: a00,a01,a02; row1: a10,a11,a12)
        # B indices: V[k0]=B[0,k0], V[k1]=B[0,k1], V[3+k0]=B[1,k0], V[3+k1]=B[1,k1]
        # W indices: W[2*k0]=C[0,k0], W[2*k0+1]=C[1,k0], W[2*k1]=C[0,k1], W[2*k1+1]=C[1,k1]
        
        # m1 = (a00+a11)(b00+b11) -> C[0,k0] += m1, C[1,k1] += m1
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[3+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00 -> C[1,k0] += m2, C[1,k1] -= m2
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11) -> C[0,k1] += m3, C[1,k1] += m3
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[3+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00) -> C[0,k0] += m4, C[1,k0] += m4
        t = col_offset
        U[4, t] += 1
        V[3+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11 -> C[0,k0] -= m5, C[0,k1] += m5
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[3+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01) -> C[1,k1] += m6
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11) -> C[0,k0] += m7
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[3+k0, t] += 1; V[3+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 outer product terms for A[:,2] * B[2,:]
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[6+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44, got {col_offset}"
    return U.tolist(), V.tolist(), W.tolist()

def _als_decompose(rank, max_iters=200, deadline=None, init=None, tol=1e-12):
    """
    ALS for tensor decomposition. Uses vectorized operations for speed.
    """
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    if init is not None:
        U = np.array(init[0], dtype=float).copy()
        V = np.array(init[1], dtype=float).copy()
        W = np.array(init[2], dtype=float).copy()
    else:
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
    
    target = _get_target()
    
    # Precompute index mappings for efficient ALS updates
    # For U[a] where a = i*m+j:
    #   Need: sum_k U[a]*V[j*p+k]*W[k*n+i] = 1 for all k
    #   => U[a] should be lstsq solution of M_U[a] * U[a] = ones(p)
    #   where M_U[a][k, r] = V[j*p+k, r] * W[k*n+i, r]
    
    # For V[b] where b = j*p+k:
    #   Need: sum_i sum_kk U[i*m+j]*V[j*p+k]*W[kk*n+i] = 1 for all (i,kk)
    #   => V[b] should be lstsq of M_V[b] * V[b] = ones(n*p)
    #   where M_V[b][idx, r] = U[i*m+j, r] * W[kk*n+i, r]
    
    # For W[c] where c = k*n+i:
    #   Need: sum_i2 sum_j U[i2*m+j]*V[j*p+k]*W[k*n+i] = 1 for all (i2,j)
    #   => W[c] should be lstsq of M_W[c] * W[c] = ones(n*m)
    #   where M_W[c][idx, r] = U[i2*m+j, r] * V[j*p+k, r]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                M = V[j*p:(j+1)*p, :] * W[[k*n+i for k in range(p)], :]
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                # M[idx, r] where idx = i*p + kk
                M = np.zeros((n*p, rank))
                for i in range(n):
                    for kk in range(p):
                        idx = i * p + kk
                        M[idx, :] = U[i*m+j, :] * W[kk*n+i, :]
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                # M[idx, r] where idx = i2*m + j
                M = np.zeros((n*m, rank))
                for i2 in range(n):
                    for j in range(m):
                        idx = i2 * m + j
                        M[idx, :] = U[i2*m+j, :] * V[j*p+k, :]
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                except:
                    pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - target))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    if best_err < 1e-6:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _polish(U, V, W, deadline, max_iters=100, tol=1e-14):
    """
    High-precision polish using ALS with tight tolerance.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    target = _get_target()
    
    best_err = np.max(np.abs(np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr) - target))
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = V_arr[j*p:(j+1)*p, :] * W_arr[[k*n+i for k in range(p)], :]
                try:
                    U_arr[a, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                for i in range(n):
                    for kk in range(p):
                        idx = i * p + kk
                        M[idx, :] = U_arr[i*m+j, :] * W_arr[kk*n+i, :]
                try:
                    V_arr[b, :], _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                for i2 in range(n):
                    for j in range(m):
                        idx = i2 * m + j
                        M[idx, :] = U_arr[i2*m+j, :] * V_arr[j*p+k, :]
                try:
                    W_arr[c, :], _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                except:
                    pass
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(T - target))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        if err < tol:
            break
    
    if best_err < 1e-8:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Fallback: schoolbook
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
    deadline = start_time + 130  # 20s safety margin
    
    # Step 1: Build rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            best_U, best_V, best_W = U44, V44, W44
            best_rank = 44
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Step 2: Try to reduce rank via ALS with multiple restarts
    # Target ranks to try, in order of preference
    target_ranks = [42, 40, 38, 36]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        num_trials = 8
        
        for trial in range(num_trials):
            if time.time() > deadline - 30:
                break
            
            # Initialize: perturb the best known solution
            if best_rank <= 44:
                U_init = np.array(best_U, dtype=float)[:, :target_rank]
                V_init = np.array(best_V, dtype=float)[:, :target_rank]
                W_init = np.array(best_W, dtype=float)[:, :target_rank]
                scale = 0.05 * (trial + 1)
                U_init = U_init + np.random.randn(n*m, target_rank) * scale
                V_init = V_init + np.random.randn(m*p, target_rank) * scale
                W_init = W_init + np.random.randn(p*n, target_rank) * scale
                init = (U_init.tolist(), V_init.tolist(), W_init.tolist())
            else:
                init = None
            
            max_iters = 150
            result = _als_decompose(target_rank, max_iters=max_iters, deadline=deadline - 30, init=init)
            if result:
                U, V, W = result
                if _verify(U, V, W, tol=1e-6):
                    # Polish
                    if time.time() < deadline - 10:
                        U_pol, V_pol, W_pol = _polish(U, V, W, deadline - 5, max_iters=80)
                        if U_pol and _verify(U_pol, V_pol, W_pol, tol=1e-8):
                            rounded = _round_and_verify(U_pol, V_pol, W_pol)
                            if rounded:
                                best_U, best_V, best_W = rounded
                            else:
                                best_U, best_V, best_W = U_pol, V_pol, W_pol
                        else:
                            rounded = _round_and_verify(U, V, W)
                            if rounded:
                                best_U, best_V, best_W = rounded
                            else:
                                best_U, best_V, best_W = U, V, W
                    else:
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
    
    # Step 3: Final polish
    if time.time() < deadline - 5:
        U_pol, V_pol, W_pol = _polish(best_U, best_V, best_W, deadline, max_iters=60)
        if U_pol and _verify(U_pol, V_pol, W_pol, tol=1e-8):
            rounded = _round_and_verify(U_pol, V_pol, W_pol)
            if rounded:
                best_U, best_V, best_W = rounded
            else:
                best_U, best_V, best_W = U_pol, V_pol, W_pol
    
    return {"U": best_U, "V": best_V, "W": best_W}
