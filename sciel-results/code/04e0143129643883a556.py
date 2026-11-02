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

def _get_target_indices():
    """Precompute target indices for verification."""
    n, m, p = DIMENSIONS
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    return target

def _compute_error(U, V, W, target=None):
    """Compute max error of tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    if target is None:
        target = _get_target_indices()
    
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
    
    Strategy: Split B (3x8) into 4 blocks of 3x2: B = [B_0, B_1, B_2, B_3]
    Each C_t = A * B_t is a 2x3 * 3x2 = 2x2 multiplication.
    
    For each 2x3*3x2 sub-problem, use rank-11 scheme:
    - Strassen rank-7 for the 2x2 output (treating A as having 3 columns)
    - Plus 4 rank-1 terms to handle the extra dimension
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        # Block handles columns 2*block and 2*block+1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for the 2x2 * 2x2 part (A cols 0,1 times B rows 0,1)
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1   # a00
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1   # a10
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        W[2*k0+1, t] += 1  # C10
        W[2*k1+1, t] += -1  # C11
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1   # a00
        V[k1, t] += 1  # b01
        V[8+k1, t] += -1  # b11
        W[2*k1, t] += 1  # C01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1   # a11
        V[8+k0, t] += 1  # b10
        V[k0, t] += -1  # b00
        W[2*k0, t] += 1  # C00
        W[2*k0+1, t] += 1  # C10
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1   # a00
        U[1, t] += 1   # a01
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += -1  # C00
        W[2*k1, t] += 1  # C01
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1   # a10
        U[0, t] += -1  # a00
        V[k0, t] += 1  # b00
        V[k1, t] += 1  # b01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1   # a01
        U[4, t] += -1  # a11
        V[8+k0, t] += 1  # b10
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        col_offset += 1
        
        # Now the 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1   # A[ii, 2]
                V[16+k_val, t] = 1  # B[2, k_val]
                W[k_val*2+ii, t] = 1  # C[ii, k_val]
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_rank40_decomposition():
    """
    Try to build a rank-40 decomposition using algebraic structure.
    
    Key insight: The 2x3x8 multiplication can be seen as computing C = A*B where
    A is 2x3 and B is 3x8. We can use the fact that the last row of A (index 2)
    only interacts with the last row of B (index 2) through 4 scalar products.
    
    Strategy: Use a combined approach where we share some Strassen terms across
    adjacent blocks to reduce the total count.
    
    Alternative: Use the polynomial encoding. Let B's columns be coefficients of
    a polynomial b(x) in R[x]. Then C = A*B corresponds to polynomial multiplication
    with specific structure. The rank of this over R[x] may be lower.
    
    Here we try: For each pair of adjacent 2-column blocks, use a shared Strassen
    structure that handles both blocks with fewer terms.
    """
    n, m, p = 2, 3, 8
    # We'll attempt rank 40: 4 blocks x 10 terms each
    # Each 2x3x2 block needs at least rank 7 (Strassen for 2x2x2) + some for the 3rd row
    # Actually 2x3x2 has rank at least 8 (lower bound for 2x3x2 is 8)
    # So 4 blocks x 10 = 40 might work if we can share
    
    U = np.zeros((n*m, 40))
    V = np.zeros((m*p, 40))
    W = np.zeros((p*n, 40))
    
    col_offset = 0
    
    # For each block of 2 columns, use rank-10 scheme:
    # 7 Strassen terms for the 2x2x2 core (rows 0,1 of A and B)
    # 3 additional terms for the 3rd row interactions (a_i2 * b_2k)
    # But we need 4 such terms (2x2), so we need 11 per block = 44 total
    # To get 40, we need to save 4 terms across 4 blocks = 1 per block
    
    # Idea: The term a_02*b_2k and a_12*b_2k for k in {k0,k1} can be
    # combined with Strassen terms if we allow cross-block sharing.
    
    # Actually, let's try a different approach: use rank-10 per block by
    # noting that the 3rd row of A times 3rd row of B gives 4 terms,
    # but we can compute a_02*b_2k0 + a_12*b_2k1 with one term if we
    # restructure. This is tricky.
    
    # Let's just return None to indicate this needs more work
    return None

def _als_decompose(rank, max_iters=100, deadline=None, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
    else:
        U = np.random.randn(n*m, rank)
    
    if init_V is not None:
        V = np.array(init_V, dtype=float).copy()
    else:
        V = np.random.randn(m*p, rank)
    
    if init_W is not None:
        W = np.array(init_W, dtype=float).copy()
    else:
        W = np.random.randn(p*n, rank)
    
    target = _get_target_indices()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
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
                    
        # Check convergence
        err = _compute_error(U, V, W, target)
        
        if err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _hooi_decompose(rank, max_iters=50, deadline=None, init_U=None, init_V=None, init_W=None):
    """
    HOOI (Higher-Order Orthogonal Iteration) for tensor decomposition.
    Uses SVD-based updates which can be more robust than ALS.
    """
    n, m, p = DIMENSIONS
    
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
    else:
        U = np.random.randn(n*m, rank)
    
    if init_V is not None:
        V = np.array(init_V, dtype=float).copy()
    else:
        V = np.random.randn(m*p, rank)
    
    if init_W is not None:
        W = np.array(init_W, dtype=float).copy()
    else:
        W = np.random.randn(p*n, rank)
    
    target = _get_target_indices()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U using SVD
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    # Use SVD for a more stable solution
                    U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
                    # Find the rank-r factor that best matches the ones vector
                    # We want U[a,:] such that M @ U[a,:] ≈ ones(p)
                    # This is equivalent to lstsq
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
        
        if err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, max_iters=200, deadline=None, tol=1e-10):
    """
    High-precision polish pass: refine a near-solution to machine precision.
    Uses damped Newton-like updates.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = _get_target_indices()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        err = _compute_error(U_arr, V_arr, W_arr, target)
        if err < tol:
            break
        
        # Small step ALS with damping
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V_arr[b, :] * W_arr[c, :]
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    # Damped update
                    U_arr[a, :] = 0.5 * U_arr[a, :] + 0.5 * x
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
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V_arr[b, :] = 0.5 * V_arr[b, :] + 0.5 * x
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
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W_arr[c, :] = 0.5 * W_arr[c, :] + 0.5 * x
                except:
                    pass
        
        err = _compute_error(U_arr, V_arr, W_arr, target)
        if err < tol:
            break
    
    if _compute_error(U_arr, V_arr, W_arr, target) < 1e-8:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
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
    except Exception as e:
        pass
    
    # Now try to find better with ALS, starting from rank 40 downward
    if best_rank == 44:
        # Try ranks 40 through 43 with multiple strategies
        for rank in range(40, best_rank):
            if time.time() > deadline:
                break
            
            # Strategy 1: Random initialization ALS
            for trial in range(4):
                if time.time() > deadline:
                    break
                np.random.seed(trial * 1000 + rank)
                result = _als_decompose(rank, max_iters=50, deadline=deadline)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            break
                        else:
                            # Try polish
                            polished = _polish(U, V, W, max_iters=100, deadline=deadline)
                            if polished:
                                best_U, best_V, best_W = polished
                                best_rank = rank
                                break
                            else:
                                best_U, best_V, best_W = U, V, W
                                best_rank = rank
                                break
            
            if best_rank < 44:
                break
            
            # Strategy 2: Structured initialization from rank-44
            # Take first 'rank' columns of the rank-44 solution and refine
            if time.time() > deadline:
                break
            try:
                U44_arr = np.array(U44, dtype=float)
                V44_arr = np.array(V44, dtype=float)
                W44_arr = np.array(W44, dtype=float)
                
                # Initialize with first 'rank' columns (structured init)
                init_U = U44_arr[:, :rank]
                init_V = V44_arr[:, :rank]
                init_W = W44_arr[:, :rank]
                
                result = _als_decompose(rank, max_iters=50, deadline=deadline,
                                       init_U=init_U, init_V=init_V, init_W=init_W)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            break
                        else:
                            polished = _polish(U, V, W, max_iters=100, deadline=deadline)
                            if polished:
                                best_U, best_V, best_W = polished
                                best_rank = rank
                                break
                            else:
                                best_U, best_V, best_W = U, V, W
                                best_rank = rank
                                break
            except:
                pass
            
            if best_rank < 44:
                break
            
            # Strategy 3: HOOI with random init
            if time.time() > deadline:
                break
            for trial in range(2):
                if time.time() > deadline:
                    break
                np.random.seed(5000 + trial * 7 + rank)
                result = _hooi_decompose(rank, max_iters=40, deadline=deadline)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                            break
                        else:
                            polished = _polish(U, V, W, max_iters=100, deadline=deadline)
                            if polished:
                                best_U, best_V, best_W = polished
                                best_rank = rank
                                break
                            else:
                                best_U, best_V, best_W = U, V, W
                                best_rank = rank
                                break
            
            if best_rank < 44:
                break
    
    # Final polish on best solution
    if best_rank < schoolbook_rank and time.time() < deadline - 10:
        try:
            polished = _polish(best_U, best_V, best_W, max_iters=200, deadline=deadline)
            if polished:
                best_U, best_V, best_W = polished
                # Try rounding again
                rounded = _round_and_verify(best_U, best_V, best_W)
                if rounded:
                    best_U, best_V, best_W = rounded
        except:
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
