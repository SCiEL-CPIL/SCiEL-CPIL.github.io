"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposition
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    err = np.max(np.abs(T - target))
    return err <= tol

def _verify_detailed(U, V, W, tol=1e-9):
    """Return max error for debugging."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
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
    Split into 4 blocks of 2x3 * 3x2, each using Strassen-7 + 4 outer products.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for 2x2*2x2 (A cols 0,1 × B rows 0,1)
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 outer product terms for A[:,2] * B[2,:]
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44
    return U.tolist(), V.tolist(), W.tolist()

def _try_remove_term(U, V, W, term_idx, deadline):
    """
    Try removing term term_idx and see if we can adjust remaining terms
    to still satisfy the decomposition. Uses least-squares on the residual.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Remove the term
    keep = [i for i in range(rank) if i != term_idx]
    U_keep = U_arr[:, keep]
    V_keep = V_arr[:, keep]
    W_keep = W_arr[:, keep]
    
    # Compute what the kept terms give
    T_kept = np.einsum('ar,br,cr->abc', U_keep, V_keep, W_keep)
    
    # Target
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    residual = target - T_kept
    
    # The removed term was: U[:,term_idx] ⊗ V[:,term_idx] ⊗ W[:,term_idx]
    # We need: sum_{r in keep} delta_U[r] ⊗ V_keep[:,r] ⊗ W_keep[:,r] 
    #           + sum_{r in keep} U_keep[:,r] ⊗ delta_V[r] ⊗ W_keep[:,r]
    #           + sum_{r in keep} U_keep[:,r] ⊗ V_keep[:,r] ⊗ delta_W[r]
    #         = residual
    # This is complex. Instead, just check if residual is close to zero.
    err = np.max(np.abs(residual))
    if err < 1e-10:
        # Perfect - just drop the term
        return U_keep.tolist(), V_keep.tolist(), W_keep.tolist()
    
    # Try to absorb into existing terms via small adjustments
    # For each kept term r, we can adjust U_keep[:,r], V_keep[:,r], W_keep[:,r]
    # Linearize: for small delta, the change in T is approximately:
    # dT = sum_r (dU_r ⊗ V_r ⊗ W_r + U_r ⊗ dV_r ⊗ W_r + U_r ⊗ V_r ⊗ dW_r)
    # This is a linear system in the deltas.
    
    num_keep = len(keep)
    # Unknowns: num_keep * (n*m + m*p + p*n)
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    total_unknowns = num_keep * (dim_u + dim_v + dim_w)
    
    # Build the linear system: A * x = residual.flatten()
    # This is too large for direct solve, use iterative approach instead.
    # Skip for now - just return None if not exact.
    
    return None

def _als_decompose_fast(rank, max_iters=100, deadline=None, init=None):
    """
    Fast ALS for tensor decomposition using vectorized operations.
    """
    n, m, p = DIMENSIONS
    
    if init is not None:
        U = np.array(init[0], dtype=float)
        V = np.array(init[1], dtype=float)
        W = np.array(init[2], dtype=float)
        if U.shape != (n*m, rank):
            U = U[:, :rank]
        if V.shape != (m*p, rank):
            V = V[:, :rank]
        if W.shape != (p*n, rank):
            W = W[:, :rank]
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve M @ x = ones where M[k,:] = V[j*p+k,:] * W[k*n+i,:]
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - target))
        if err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, deadline, max_iters=50):
    """
    High-precision polish: use Newton-like iterations to drive error to machine precision.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current error
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(T - target))
        if err < 1e-14:
            break
        
        # Do one ALS step for refinement
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V_arr[b, :] * W_arr[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_arr[a, :] = x
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U_arr[a, :] * W_arr[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V_arr[b, :] = x
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U_arr[a, :] * V_arr[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W_arr[c, :] = x
                except:
                    pass
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

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
    deadline = start_time + 145  # 5s safety margin
    
    # Step 1: Build rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            best_U, best_V, best_W = U44, V44, W44
            best_rank = 44
        else:
            # Try rounding
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Step 2: Try to reduce rank by removing terms
    # For rank-44, try removing each term one at a time
    if best_rank == 44:
        U_arr = np.array(best_U, dtype=float)
        V_arr = np.array(best_V, dtype=float)
        W_arr = np.array(best_W, dtype=float)
        
        # Try removing each of the 44 terms
        for term_idx in range(44):
            if time.time() > deadline:
                break
            
            result = _try_remove_term(best_U, best_V, best_W, term_idx, deadline)
            if result:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, tol=1e-6):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = 43
                    break
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Step 3: Try ALS for lower ranks
    if best_rank > 40:
        target_ranks = [42, 41, 40, 39, 38, 37, 36]
        
        for rank in target_ranks:
            if time.time() > deadline:
                break
            
            # Try with initialization from current best (truncated)
            for trial in range(3):
                if time.time() > deadline:
                    break
                
                if rank < best_rank:
                    init = (best_U[:rank], best_V[:rank], best_W[:rank])
                else:
                    init = None
                
                result = _als_decompose_fast(rank, max_iters=80, deadline=deadline, init=init)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        # Polish
                        U_pol, V_pol, W_pol = _polish(U, V, W, deadline, max_iters=20)
                        if _verify(U_pol, V_pol, W_pol, tol=1e-8):
                            rounded = _round_and_verify(U_pol, V_pol, W_pol)
                            if rounded:
                                best_U, best_V, best_W = rounded
                            else:
                                best_U, best_V, best_W = U_pol, V_pol, W_pol
                            best_rank = rank
                            break
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                            break
    
    # Step 4: Final polish and rounding
    if time.time() < deadline - 5:
        U_pol, V_pol, W_pol = _polish(best_U, best_V, best_W, deadline, max_iters=30)
        if _verify(U_pol, V_pol, W_pol, tol=1e-8):
            rounded = _round_and_verify(U_pol, V_pol, W_pol)
            if rounded:
                best_U, best_V, best_W = rounded
            else:
                best_U, best_V, best_W = U_pol, V_pol, W_pol
    
    return {"U": best_U, "V": best_V, "W": best_W}
