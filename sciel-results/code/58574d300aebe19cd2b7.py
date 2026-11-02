"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
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
    
    # Build the full tensor T = sum_r U[:,r] * V[:,r] * W[:,r]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Check target entries
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
    
    # Check non-target entries are zero
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    
    Strategy: Split B (3x8) into 4 blocks of 3x2: B = [B_0, B_1, B_2, B_3]
    Each C_t = A * B_t is a 2x3 * 3x2 = 2x2 multiplication.
    
    For each 2x3*3x2 sub-problem, use rank-11 scheme:
    - Strassen rank-7 for the 2x2 output (treating A as having 3 columns)
    - Plus 4 rank-1 terms to handle the extra dimension
    
    For M_{2,3,2}:
    A is 2x3, B is 3x2, C is 2x2.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Write A = [A_left | A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x1 (col 2).
    Write B = [B_top; B_bottom] where B_top is 2x2 (rows 0,1) and B_bottom is 1x2 (row 2).
    
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top is 2x2 * 2x2, which has rank 7 (Strassen).
    A_right * B_bottom is 2x1 * 1x2, which requires 4 rank-1 terms.
    Total: 7 + 4 = 11 per sub-block.
    4 sub-blocks: 4 * 11 = 44.
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
        # A_left: rows 0,1 of A, cols 0,1 -> indices i*3+j for i in {0,1}, j in {0,1}
        # B_top: rows 0,1 of B, cols k0,k1 -> indices j*8+k for j in {0,1}, k in {k0,k1}
        # C: 2x2 output, entries C[i, k0] and C[i, k1] for i in {0,1}
        # W index: k*n+i = k*2+i
        
        # Strassen products for 2x2*2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # a00 = A[0,0] -> U index 0*3+0 = 0
        # a01 = A[0,1] -> U index 0*3+1 = 1
        # a10 = A[1,0] -> U index 1*3+0 = 3
        # a11 = A[1,1] -> U index 1*3+1 = 4
        
        # b00 = B[0,k0] -> V index 0*8+k0 = k0
        # b01 = B[0,k1] -> V index 0*8+k1 = k1
        # b10 = B[1,k0] -> V index 1*8+k0 = 8+k0
        # b11 = B[1,k1] -> V index 1*8+k1 = 8+k1
        
        # C00: i=0,k=k0 -> W index k0*2+0 = 2*k0
        # C01: i=0,k=k1 -> W index k1*2+0 = 2*k1
        # C10: i=1,k=k0 -> W index k0*2+1 = 2*k0+1
        # C11: i=1,k=k1 -> W index k1*2+1 = 2*k1+1
        
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
        # A_right = A[:, 2] -> U indices 0*3+2=2 and 1*3+2=5
        # B_bottom = B[2, :] -> V indices 2*8+k = 16+k for k in {k0, k1}
        
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                U[ii*3+2, t] = 1  # A_right[ii]
                V[16+k0+kk, t] = 1  # B_bottom[kk]
                W[(k0+kk)*2+ii, t] = 1  # C[ii, k0+kk]
                col_offset += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, target_rank, max_time=30):
    """
    Try to reduce the rank by finding linear dependencies among the columns.
    Uses iterative elimination with verification.
    """
    start_time = time.time()
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    current_rank = U.shape[1]
    
    # Try to eliminate columns one at a time
    for _ in range(current_rank - target_rank):
        if time.time() - start_time > max_time:
            break
            
        best_elimination = None
        best_residual = float('inf')
        
        # Try eliminating each column
        for col_to_remove in range(current_rank):
            # Check if this column is linearly dependent
            # Simple check: see if we can express this column as a combination of others
            # For now, just try removing it and see if we can adjust others
            
            U_test = np.delete(U, col_to_remove, axis=1)
            V_test = np.delete(V, col_to_remove, axis=1)
            W_test = np.delete(W, col_to_remove, axis=1)
            
            # Verify if the reduced decomposition still works
            if _verify(U_test.tolist(), V_test.tolist(), W_test.tolist(), tol=1e-6):
                best_elimination = col_to_remove
                best_residual = 0.0
                break
        
        if best_elimination is not None:
            U = np.delete(U, best_elimination, axis=1)
            V = np.delete(V, best_elimination, axis=1)
            W = np.delete(W, best_elimination, axis=1)
            current_rank -= 1
        else:
            break
    
    if current_rank < target_rank:
        return U, V, W, current_rank
    
    return U, V, W, current_rank

def _als_decompose(U0, V0, W0, target_rank, max_iter=200, tol=1e-8, max_time=60):
    """
    ALS-based decomposition to try to find a lower rank representation.
    Start from the rank-44 solution and try to project to lower rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.array(U0, dtype=float).copy()
    V = np.array(V0, dtype=float).copy()
    W = np.array(W0, dtype=float).copy()
    current_rank = U.shape[1]
    
    # Project to target rank by truncation (keep first target_rank columns)
    # This is a valid (though not optimal) way to reduce rank for ALS initialization
    if current_rank > target_rank:
        U = U[:, :target_rank].copy()
        V = V[:, :target_rank].copy()
        W = W[:, :target_rank].copy()
        current_rank = target_rank
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        # Update U: for each column r, minimize ||T - sum_s U_s V_s W_s||_F
        # This is equivalent to: U[:,r] = argmin ||T - sum_s != r U_s V_s W_s - U[:,r] V[:,r] W[:,r]||
        # Which gives: U[:,r] = (T - sum_s != r U_s V_s W_s) : (V[:,r] W[:,r])
        # where : is the tensor contraction
        
        for r in range(current_rank):
            # Compute residual without term r
            T_res = np.zeros((nm, mp, pn))
            for s in range(current_rank):
                if s != r:
                    T_res += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
            
            # Target tensor
            T_target = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Residual that U[:,r] should match
            residual = T_target - T_res
            
            # Solve: minimize ||residual - U[:,r] V[:,r] W[:,r]||_F
            # This is: U[:,r] = argmin ||residual - u v w||_F
            # = (residual : (v w)) / (v^T w^T) ... not quite
            # Actually: sum_{a,b,c} residual[a,b,c] * v[b] * w[c] = sum_a U[a,r] * (sum_{b,c} v[b] w[c] ... no
            # The objective is: sum_{a,b,c} (residual[a,b,c] - U[a,r]*V[b,r]*W[c,r])^2
            # d/dU[a,r] = -2 sum_{b,c} residual[a,b,c] V[b,r] W[c,r] + 2 U[a,r] sum_{b,c} V[b,r]^2 W[c,r]^2
            # So: U[a,r] = sum_{b,c} residual[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
            
            denom = np.sum(V[:,r]**2 * W[:,r]**2)
            if abs(denom) < 1e-15:
                continue
            U[:,r] = np.einsum('abc,b,c->a', residual, V[:,r], W[:,r]) / denom
        
        # Update V
        for r in range(current_rank):
            T_res = np.zeros((nm, mp, pn))
            for s in range(current_rank):
                if s != r:
                    T_res += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
            
            T_target = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            residual = T_target - T_res
            
            denom = np.sum(U[:,r]**2 * W[:,r]**2)
            if abs(denom) < 1e-15:
                continue
            V[:,r] = np.einsum('abc,a,c->b', residual, U[:,r], W[:,r]) / denom
        
        # Update W
        for r in range(current_rank):
            T_res = np.zeros((nm, mp, pn))
            for s in range(current_rank):
                if s != r:
                    T_res += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
            
            T_target = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            residual = T_target - T_res
            
            denom = np.sum(U[:,r]**2 * V[:,r]**2)
            if abs(denom) < 1e-15:
                continue
            W[:,r] = np.einsum('abc,a,b->c', residual, U[:,r], V[:,r]) / denom
        
        # Check convergence
        T_check = np.zeros((nm, mp, pn))
        for r in range(current_rank):
            T_check += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        err = np.linalg.norm(T_check - T_target)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with the known rank-44 construction
    U, V, W = _build_rank44_decomposition()
    best_rank = 44
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify the initial construction
    if not _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
        # Fallback to schoolbook if something went wrong
        n, m, p = DIMENSIONS
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce the rank via ALS
    if time.time() < deadline:
        for target_rank in [40, 38, 36, 34, 32, 30, 28]:
            if time.time() > deadline - 10:
                break
            
            try:
                U_als, V_als, W_als, err = _als_decompose(
                    U, V, W, target_rank, max_iter=50, tol=1e-8, max_time=10
                )
                
                # Round to nearest half-integer and verify
                U_rounded = np.round(U_als * 2) / 2
                V_rounded = np.round(V_als * 2) / 2
                W_rounded = np.round(W_als * 2) / 2
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
            except Exception as e:
                # If ALS fails, break and keep the best known (Rank 44)
                break
    
    # Try simple column elimination
    if time.time() < deadline:
        try:
            U_red, V_red, W_red, new_rank = _try_reduce_rank(best_U, best_V, best_W, best_rank - 1, max_time=5)
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U_red, V_red, W_red
        except Exception:
            pass
    
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        arr_rounded = np.round(arr * 2) / 2
        return arr_rounded.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
