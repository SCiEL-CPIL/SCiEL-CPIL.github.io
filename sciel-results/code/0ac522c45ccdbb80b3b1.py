import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def try_rank(r, max_iters=100, tol=1e-8, num_restarts=20, deadline=None):
        """
        Attempts to find a decomposition of rank r using vectorized ALS.
        """
        best_result = None
        
        for restart in range(num_restarts):
            if deadline and time.time() > deadline:
                break
                
            np.random.seed(12345 + restart)
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return best_result
                    
                # Update U
                # For each i, solve: sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k]
                # Let M_U be matrix (dim_v*dim_w, r) where M_U[(j,k), c] = V[j,c] W[k,c]
                # Then U[i,:] = pinv(M_U) @ T[i,:,:].flatten()
                
                # Construct M_U
                # V is (dim_v, r), W is (dim_w, r)
                # We want outer product element-wise for each c?
                # M_U[j*dim_w + k, c] = V[j, c] * W[k, c]
                # This is equivalent to:
                # M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v*dim_w, r)
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                # Compute pinv once
                try:
                    M_U_pinv = np.linalg.pinv(M_U)
                except np.linalg.LinAlgError:
                    continue
                # Update all U rows
                # T is (dim_u, dim_v, dim_w). Reshape to (dim_u, dim_v*dim_w)
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                U_new = (M_U_pinv @ T_flat.T).T # (dim_u, r)
                
                # Update V
                # For each j, solve: sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k]
                # Variables are V[j,c].
                # Equation: sum_c V[j,c] (U[i,c] W[k,c]) = T[i,j,k]
                # Let M_V be matrix (dim_u*dim_w, r) where M_V[(i,k), c] = U_new[i,c] W[k,c]
                # Note: using updated U
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    M_V_pinv = np.linalg.pinv(M_V)
                except np.linalg.LinAlgError:
                    continue
                # T reshaped for V update: for fixed j, we need T[:, j, :] flattened
                # T is (n*m, m*p, p*n). 
                # Index I = i*m+j, J = j*p+k, K = k*n+i.
                # This indexing is tricky for simple slicing.
                # Let's stick to the explicit construction for RHS to avoid indexing errors, 
                # or pre-compute the mapping.
                
                # Actually, let's pre-compute the indices for T slices.
                # For V update (fixed j):
                # We need b_j of length dim_u * dim_w.
                # b_j[(i,k)] = T[I, J, K] where I=i*m+j, J=j*p+k, K=k*n+i.
                # This is a permutation of T's elements.
                
                # Let's pre-compute these flattened RHS vectors for all i, j, k at the start.
                # This avoids doing it inside the loop.
                
                # Re-initialize pre-computed RHS
                # This is expensive if done every restart, but cheap if done once.
                # Let's do it outside.
                
                # Wait, the code structure below will move this.
                
                U = U_new
                # Placeholder for V, W updates to be implemented in the edit
                # For now, just U update logic to show structure.
                # The actual edit will replace the whole function.
                
        return None

    # Pre-compute RHS vectors for ALS
    # For U update (fixed i): b_i = T[i, :, :].flatten()
    # For V update (fixed j): b_j[(i,k)] = T[i*m+j, j*p+k, k*n+i]
    # For W update (fixed k): b_k[(i,j)] = T[i*m+j, j*p+k, k*n+i]
    
    RHS_U = T.reshape(dim_u, dim_v * dim_w) # (dim_u, dim_v*dim_w)
    
    RHS_V = np.zeros((dim_v, dim_u * dim_w))
    for j in range(dim_v):
        for i in range(dim_u):
            for k in range(dim_w):
                # I = i, J = j, K = k is NOT correct.
                # I = i_row * m + j_col? 
                # The tensor T is defined by:
                # I = i * m + j
                # J = j * p + k
                # K = k * n + i
                # Here i is in [0,n), j in [0,m), k in [0,p).
                # dim_u = n*m. Index I ranges 0..n*m-1.
                # dim_v = m*p. Index J ranges 0..m*p-1.
                # dim_w = p*n. Index K ranges 0..p*n-1.
                
                # For V update, we fix the "middle" index J?
                # No, V corresponds to the dimension m*p.
                # The variable is V[J, c].
                # The equation is T[I, J, K] = sum_c U[I,c] V[J,c] W[K,c].
                # For fixed J, we solve for V[J, :].
                # The unknowns are V[J, 0..r-1].
                # The equations are for all I, K.
                # Coefficient of V[J, c] is U[I, c] * W[K, c].
                # RHS is T[I, J, K].
                # So we need to flatten T[:, J, :] into a vector of length dim_u * dim_w.
                # But T is stored as (dim_u, dim_v, dim_w).
                # So T[:, J, :] is a slice of shape (dim_u, dim_w).
                # We just need to flatten it.
                pass

    # Let's refine the pre-computation.
    # T is (dim_u, dim_v, dim_w).
    # For U update (fix I): RHS is T[I, :, :].flatten()
    # For V update (fix J): RHS is T[:, J, :].flatten()
    # For W update (fix K): RHS is T[:, :, K].flatten()
    
    # This is much simpler! The indexing I=i*m+j etc was for constructing T.
    # Once T is constructed, the ALS operates on the matrix slices.
    
    RHS_U = np.array([T[i, :, :].flatten() for i in range(dim_u)]) # (dim_u, dim_v*dim_w)
    RHS_V = np.array([T[:, j, :].flatten() for j in range(dim_v)]) # (dim_v, dim_u*dim_w)
    RHS_W = np.array([T[:, :, k].flatten() for k in range(dim_w)]) # (dim_w, dim_u*dim_v)

    def try_rank_v2(r, max_iters=100, tol=1e-8, num_restarts=50, deadline=None):
        best_result = None
        for restart in range(num_restarts):
            if deadline and time.time() > deadline:
                break
            np.random.seed(1000 + restart)
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return best_result
                
                # Update U
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                try:
                    M_U_pinv = np.linalg.pinv(M_U)
                    U_new = (M_U_pinv @ RHS_U.T).T
                except:
                    continue
                
                # Update V
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    M_V_pinv = np.linalg.pinv(M_V)
                    V_new = (M_V_pinv @ RHS_V.T).T
                except:
                    continue
                
                # Update W
                M_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(dim_u * dim_v, r)
                try:
                    M_W_pinv = np.linalg.pinv(M_W)
                    W_new = (M_W_pinv @ RHS_W.T).T
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check error
                # T_approx[I,J,K] = sum_c U[I,c] V[J,c] W[K,c]
                # Efficient check:
                # For a few random indices or all? All is 900 entries, cheap.
                # T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
                # err = np.linalg.norm(T_approx - T)
                
                # Faster: check if close to half-integers and verify exactly.
                # This is the key.
                # If the solution is close to half-integers, snap and verify.
                # This avoids expensive norm calculation every step if we are not close.
                # But we need to know when to check.
                # Let's check every 5 iterations.
                if it % 5 == 0:
                    U_h = np.round(U * 2) / 2
                    V_h = np.round(V * 2) / 2
                    W_h = np.round(W * 2) / 2
                    
                    # Check if snapping changed values significantly
                    # If not, verify exactness
                    if np.allclose(U, U_h) and np.allclose(V, V_h) and np.allclose(W, W_h):
                        T_approx = np.einsum('ic,jc,kc->ijk', U_h, V_h, W_h)
                        if np.allclose(T_approx, T, atol=1e-9):
                            return {
                                "U": U_h.tolist(),
                                "V": V_h.tolist(),
                                "W": W_h.tolist()
                            }
                            
        return None

    deadline = time.time() + 110 # Leave 10s margin
    
    # Try rank 29
    res = try_rank_v2(29, max_iters=200, num_restarts=100, deadline=deadline)
    if res:
        return res
        
    # Fallback to schoolbook (rank 30)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
