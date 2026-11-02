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
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
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
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _get_target_indices(n, m, p):
    """Precompute target indices for verification."""
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    return target

def _compute_error(U, V, W, target):
    """Compute max error of tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    target_err = np.max(np.abs(T[target] - 1.0))
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max(target_err, non_target_err)

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

def _find_rank10_2x3x2(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x3x2 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 3, 2
    rank = 10
    target = _get_target_indices(n, m, p)
    
    # Flatten the target tensor into a vector of non-zero entries
    # T[a, b, c] = 1 for a=i*m+j, b=j*p+k, c=k*n+i
    
    # We will use ALS to solve for U, V, W
    # U: 6x10, V: 6x10, W: 4x10
    
    for trial in range(20):
        if time.time() > deadline:
            return None
            
        np.random.seed(trial)
        U = np.random.randn(6, rank)
        V = np.random.randn(6, rank)
        W = np.random.randn(4, rank)
        
        for it in range(100):
            if time.time() > deadline:
                return None
                
            # Update U
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    M = np.zeros((p, rank))
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        M[k, :] = V[b, :] * W[c, :]
                    try:
                        x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
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
                        x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
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
                        x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                        W[c, :] = x
                    except:
                        pass
                        
            err = _compute_error(U, V, W, target)
            if err < 1e-8:
                # Verify and round
                rounded = _round_and_verify(U.tolist(), V.tolist(), W.tolist())
                if rounded:
                    return rounded
                else:
                    # Return the float solution, verify will be done later
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
    deadline = start_time + 140
    
    # First, try to find a rank-40 solution using a rank-10 2x3x2 block
    # Deadline for this search: 30 seconds
    sub_deadline = start_time + 30
    rank10_block = _find_rank10_2x3x2(sub_deadline)
    
    if rank10_block:
        U_block, V_block, W_block = rank10_block
        # Verify the block
        target_block = _get_target_indices(2, 3, 2)
        U_b_arr = np.array(U_block)
        V_b_arr = np.array(V_block)
        W_b_arr = np.array(W_block)
        T_b = np.zeros((6, 6, 4))
        for r in range(10):
            T_b += np.einsum('a,b,c->abc', U_b_arr[:,r], V_b_arr[:,r], W_b_arr[:,r])
        err_b = max(np.max(np.abs(T_b[target_block] - 1.0)), np.max(np.abs(T_b[~target_block])))
        
        if err_b < 1e-6:
            # Build full rank-40 solution
            U_full = np.zeros((6, 40))
            V_full = np.zeros((24, 40))
            W_full = np.zeros((16, 40))
            
            for block in range(4):
                col_offset = block * 10
                k0 = 2 * block
                k1 = 2 * block + 1
                
                # U is 6x10, same for all blocks
                U_full[:, col_offset:col_offset+10] = U_b_arr
                
                # V is 6x10 for the block, needs to be placed in 24x40
                # V indices for block:
                # j=0: k0, k1 -> V[k0, :], V[k1, :]
                # j=1: 8+k0, 8+k1 -> V[8+k0, :], V[8+k1, :]
                # j=2: 16+k0, 16+k1 -> V[16+k0, :], V[16+k1, :]
                V_full[k0, col_offset:col_offset+10] = V_b_arr[0, :]
                V_full[k1, col_offset:col_offset+10] = V_b_arr[1, :]
                V_full[8+k0, col_offset:col_offset+10] = V_b_arr[2, :]
                V_full[8+k1, col_offset:col_offset+10] = V_b_arr[3, :]
                V_full[16+k0, col_offset:col_offset+10] = V_b_arr[4, :]
                V_full[16+k1, col_offset:col_offset+10] = V_b_arr[5, :]
                
                # W is 4x10 for the block, needs to be placed in 16x40
                # W indices for block:
                # k=k0: i=0,1 -> W[k0*2+0, :], W[k0*2+1, :]
                # k=k1: i=0,1 -> W[k1*2+0, :], W[k1*2+1, :]
                W_full[2*k0+0, col_offset:col_offset+10] = W_b_arr[0, :]
                W_full[2*k0+1, col_offset:col_offset+10] = W_b_arr[1, :]
                W_full[2*k1+0, col_offset:col_offset+10] = W_b_arr[2, :]
                W_full[2*k1+1, col_offset:col_offset+10] = W_b_arr[3, :]
                
            # Verify full solution
            if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), tol=1e-6):
                best_U, best_V, best_W = U_full.tolist(), V_full.tolist(), W_full.tolist()
                best_rank = 40
    
    # If rank 40 not found, try rank 44
    if best_rank > 44:
        try:
            U44, V44, W44 = _build_rank44_decomposition()
            if _verify(U44, V44, W44, tol=1e-6):
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
        except:
            pass
            
    # Final polish
    if best_rank < schoolbook_rank and time.time() < deadline - 5:
        try:
            # Simple polish: re-run ALS for a few iterations on the best solution
            U_arr = np.array(best_U)
            V_arr = np.array(best_V)
            W_arr = np.array(best_W)
            rank = len(U_arr[0])
            target = _get_target_indices(n, m, p)
            
            for it in range(50):
                if time.time() > deadline:
                    break
                # Update U
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        M = np.zeros((p, rank))
                        for k in range(p):
                            b = j * p + k
                            c = k * n + i
                            M[k, :] = V_arr[b, :] * W_arr[c, :]
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_arr[a, :] = 0.5 * U_arr[a, :] + 0.5 * x
                        
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
                                M[idx, :] = U_arr[a, :] * W_arr[c, :]
                                idx += 1
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                        V_arr[b, :] = 0.5 * V_arr[b, :] + 0.5 * x
                        
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
                                M[idx, :] = U_arr[a, :] * V_arr[b, :]
                                idx += 1
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                        W_arr[c, :] = 0.5 * W_arr[c, :] + 0.5 * x
                        
            if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), tol=1e-6):
                best_U, best_V, best_W = U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
        except:
            pass
            
    return {"U": best_U, "V": best_V, "W": best_W}
