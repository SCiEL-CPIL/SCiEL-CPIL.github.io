import numpy as np
import time

def construct():
    n, m, p = 2, 2, 11
    deadline = time.time() + 110  # safety margin before 120s kill

    # Build the rank-39 Strassen construction
    U, V, W = _build_strassen(n, m, p)
    best_rank = 39
    best_U, best_V, best_W = U, V, W

    # Convert to numpy for efficient operations
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build target tensor T[i,j,k] = delta_{i,k} for C = A*B
    # T is a 3-tensor of shape (n*m, m*p, p*n)
    # T[a_idx, b_idx, c_idx] = 1 if a_idx and c_idx correspond to same row/col pattern
    # a_idx = i*m + j, b_idx = j*p + k, c_idx = k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        """Check if U,V,W exactly decompose T."""
        for r in range(U_arr.shape[1]):
            if np.any(np.abs(U_arr[:, r]) > tol) and np.any(np.abs(V_arr[:, r]) > tol) and np.any(np.abs(W_arr[:, r]) > tol):
                # Check: sum over r of U[:,r] * V[:,r] * W[:,r] == T
                pass
        # Efficient check: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] should equal T[a,b,c]
        # Build matrix M[a,b] = sum_r U[a,r]*V[b,r], then check M[a,b]*W[c,r]... 
        # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed c: sum_r (U*V^T)[a,b] * W[c,r] ... no
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r (U[:,r] * V[:,r])_a,b * W[c,r]  -- no, that's elementwise
        # Let's do: for each r, the outer product U[:,r] @ V[:,r].T @ W[c,r] contributes
        # Actually T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Group: for each c, T[:,:,c] = sum_r (U[:,r] * V[:,r].T) * W[c,r]
        # That's not quite right either. Let me think...
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed (a,b): T[a,b,:] = sum_r (U[a,r] * V[b,r]) * W[:,r].T ... 
        # = sum_r W[:,r] * (U[a,r]*V[b,r])
        # So T[a,b,:] = W.T @ (U[a,:]*V[b,:])
        # Let's verify this way
        for a in range(nm):
            for b in range(mp):
                expected = T[a, b, :]
                actual = np.array([sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(U_arr.shape[1])) for c in range(pn)])
                if np.max(np.abs(expected - actual)) > tol:
                    return False
        return True

    # More efficient verification using matrix operations
    def verify_fast(U_arr, V_arr, W_arr, tol=1e-9):
        rank = U_arr.shape[1]
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For each c: T[:,:,c] = sum_r W[c,r] * (U[:,r] @ V[:,r].T)
        # Precompute UV_products
        # Actually: T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
        # Let's build: for each r, M_r = W[:,r] @ (U[:,r] @ V[:,r].T) ... no
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r W[c,r] * (U[:,r] * V[:,r].T)[a,b]  -- no, U[:,r]*V[:,r].T is a matrix
        # Let's just do it directly but vectorized per c
        for c in range(pn):
            # T[:,:,c] = sum_r W[c,r] * (U[:,r] @ V[:,r].T)  -- this is wrong
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: T[:,:,c][a,b] = sum_r W[c,r] * U[a,r] * V[b,r]
            # = sum_r W[c,r] * (U[:,r] @ V[:,r].T)[a,b]  -- NO, that's not right
            # U[:,r] is a vector, V[:,r] is a vector, their outer product is U[:,r] @ V[:,r].T
            # So T[:,:,c] = sum_r W[c,r] * (U[:,r] @ V[:,r].T)
            # But U[:,r] @ V[:,r].T is an (nm x mp) matrix
            # This is correct!
            accumulated = np.zeros((nm, mp))
            for r in range(rank):
                accumulated += W_arr[c, r] * np.outer(U_arr[:, r], V_arr[:, r])
            if np.max(np.abs(accumulated - T[:, :, c])) > tol:
                return False
        return True

    # Try greedy rank reduction
    # Strategy: try to remove each column and see if we can compensate
    # For a given set of columns to keep, check if the tensor is in their span
    
    def tensor_in_span(U_keep, V_keep, W_keep, tol=1e-9):
        """Check if T can be represented with only the kept columns."""
        # T[a,b,c] = sum_{r in kept} U[a,r]*V[b,r]*W[c,r]
        # This is equivalent to: for each c, T[:,:,c] must be in span of {W[c,r]*outer(U[:,r],V[:,r]) for r in kept}
        # But that's not quite right. Let me think differently.
        # The set of all tensors representable with these columns is:
        # {sum_{r in kept} alpha_r * (U[:,r] @ V[:,r].T @ W[:,r].T) ... no
        # Actually each column r defines a rank-1 tensor: U[:,r] @ V[:,r].T @ W[:,r].T (3-tensor outer)
        # We need T = sum_{r in kept} U[:,r] @ V[:,r].T @ W[:,r].T
        # Flatten: vec(T) = sum_{r in kept} vec(U[:,r] @ V[:,r].T @ W[:,r].T)
        # This is a linear system: vec(T) = M @ [1,1,...,1] where M has columns vec(outer3(U[:,r],V[:,r],W[:,r]))
        # But we want to check if there exist coefficients (all 1) that work, OR
        # if we can scale the columns.
        # Actually the question is: can we find scalars s_r such that sum_r s_r * (U[:,r]@V[:,r].T@W[:,r].T) = T?
        # No -- the decomposition requires the specific coefficients to be 1 (or whatever they are in U,V,W)
        # Wait, no. In a tensor decomposition, we have T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # The coefficients ARE the entries of U,V,W. So if we keep columns r1,...,rk,
        # we need sum_{i} U[:,ri] ⊗ V[:,ri] ⊗ W[:,ri] = T
        # This is a fixed question: does this sum equal T?
        
        rank = U_keep.shape[1]
        for c in range(pn):
            accumulated = np.zeros((nm, mp))
            for r in range(rank):
                accumulated += W_keep[c, r] * np.outer(U_keep[:, r], V_keep[:, r])
            if np.max(np.abs(accumulated - T[:, :, c])) > tol:
                return False
        return True

    # Greedy: try removing each column, check if still valid
    # If not directly, try adjusting remaining columns slightly
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = 39
    
    # Build flattened tensor
    T_flat = T.reshape(-1)
    nm_mp_pn = nm * mp * pn
    
    def build_flattened_columns(U_arr, V_arr, W_arr):
        """Build matrix where column r is vec(U[:,r] @ V[:,r].T @ W[:,r].T) flattened as 3-tensor."""
        # vec of 3-tensor: index (a,b,c) -> a*mp*pn + b*pn + c
        rank = U_arr.shape[1]
        M = np.zeros((nm_mp_pn, rank))
        for r in range(rank):
            # 3-tensor: T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
            # Flatten: T_r[a,b,c] -> M[a*mp*pn + b*pn + c, r]
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        M[a * mp * pn + b * pn + c, r] = U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
        return M

    # Try removing columns one at a time
    # For each column r to remove, check if T is in span of remaining columns
    # i.e., solve M_remaining @ x = T_flat for x, and check if solution exists
    # But we need x to be such that the decomposition is "nice" (integer/half-integer)
    # For now, just check if T is in the span (linear dependency)
    
    removed = set()
    for attempt in range(20):  # try up to 20 removals
        if time.time() > deadline - 10:
            break
        found_removal = False
        for r in range(current_rank):
            if r in removed:
                continue
            # Try removing column r
            keep = [i for i in range(current_rank) if i != r and i not in removed]
            if len(keep) == 0:
                continue
            U_keep = current_U[:, keep]
            V_keep = current_V[:, keep]
            W_keep = current_W[:, keep]
            
            # Check if T is in span of these columns
            # Build flattened matrix
            M = build_flattened_columns(U_keep, V_keep, W_keep)
            # Check if T_flat is in column space of M
            # Use least squares: solve M @ x = T_flat
            try:
                x, residuals, rank_m, sv = np.linalg.lstsq(M, T_flat, rcond=1e-10)
                # Check if residual is small
                if residuals[0] < 1e-8 * np.linalg.norm(T_flat):
                    # T is in span! But we need to check if the solution x gives valid coefficients
                    # The decomposition would be: T = sum_{i in keep} x[i] * (U[:,keep[i]] @ V[:,keep[i]].T @ W[:,keep[i]].T)
                    # This means we need to scale columns: new_U[:,i] = x[i] * U[:,keep[i]], etc.
                    # But then the entries might not be integer/half-integer
                    # For now, accept it and update
                    new_U = np.zeros((nm, len(keep)))
                    new_V = np.zeros((mp, len(keep)))
                    new_W = np.zeros((pn, len(keep)))
                    for i, ki in enumerate(keep):
                        new_U[:, i] = x[i] * U_keep[:, i]
                        new_V[:, i] = x[i] * V_keep[:, i]
                        new_W[:, i] = x[i] * W_keep[:, i]
                    
                    # Verify
                    if verify_fast(new_U, new_V, new_W, tol=1e-8):
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank = len(keep)
                        removed.add(r)
                        found_removal = True
                        break
            except:
                continue
        
        if not found_removal:
            break
    
    # Final verification
    if verify_fast(current_U, current_V, current_W, tol=1e-8):
        best_U = current_U
        best_V = current_V
        best_W = current_W
        best_rank = current_rank

    # Convert back to list format
    U_out = [[float(best_U[i, r]) for r in range(best_rank)] for i in range(nm)]
    V_out = [[float(best_V[i, r]) for r in range(best_rank)] for i in range(mp)]
    W_out = [[float(best_W[i, r]) for r in range(best_rank)] for i in range(pn)]

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_strassen(n, m, p):
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    return U, V, W
