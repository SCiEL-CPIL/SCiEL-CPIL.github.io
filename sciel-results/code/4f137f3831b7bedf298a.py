"""Matrix multiplication 3,4,7 - search for rank-minimal decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 12
    dim_v = m * p  # 28
    dim_w = p * n  # 21
    
    # Construct the target tensor T
    # T[i, j, k] is not quite right.
    # The tensor is defined on U (i,j), V (j,k), W (k,i).
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # We represent T as a flat vector of size dim_u * dim_v * dim_w
    # Or better, we work with the "matricized" forms for ALS.
    
    # For ALS, we need to solve linear systems.
    # Let's create the full tensor T as a 3D array of shape (dim_u, dim_v, dim_w)
    # This is 12 * 28 * 21 = 7056 elements. Very small.
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    T_flat = T.flatten()
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=np.float64)
        V = np.zeros((dim_v, rank), dtype=np.float64)
        W = np.zeros((dim_w, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        # Check if entries are close to integer or half-integer
        # and if the decomposition is exact
        rank = U.shape[1]
        if rank >= best_rank:
            return False, None
        
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check if snapping changed values significantly (optional, but good for sanity)
        # Actually, we just need the final result to be exact.
        
        # Compute reconstruction
        # T_recon[u, v, w] = sum_k U[u,k] * V[v,k] * W[w,k]
        # We can do this efficiently using einsum
        T_recon = np.einsum('uk,vk,wk->uvw', U_s, V_s, W_s)
        
        # Check exact match
        if np.allclose(T_recon, T, atol=1e-9):
            # Ensure we return lists of lists as per interface? 
            # The interface says 'U', 'V', 'W' with integer/half-integer entries.
            # Usually numpy arrays are accepted if they contain floats representing these.
            # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
            # I will return numpy arrays.
            return True, {"U": U_s, "V": V_s, "W": W_s}
            
        return False, None

    def khatri_rao(A, B):
        # A is (M, R), B is (N, R)
        # Returns (M*N, R)
        # A[i, k] * B[j, k]
        # Reshape A to (M, 1, R) and B to (1, N, R)
        # Multiply and reshape to (M*N, R)
        A_reshaped = A[:, np.newaxis, :]
        B_reshaped = B[np.newaxis, :, :]
        prod = A_reshaped * B_reshaped
        return prod.reshape(-1, B.shape[1])

    def als_decompose(r, max_iters=50, tol=1e-10):
        # Initialize with random values
        # Scale initialization to match the norm of T
        # T has 84 ones. Norm is sqrt(84).
        # Factors should be roughly of magnitude 1.
        
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        W = np.random.randn(dim_w, r)
        
        # Normalize columns to prevent divergence
        for i in range(r):
            norm_u = np.linalg.norm(U[:, i])
            norm_v = np.linalg.norm(V[:, i])
            norm_w = np.linalg.norm(W[:, i])
            total_norm = norm_u * norm_v * norm_w
            if total_norm > 1e-10:
                # Distribute the norm
                # Just normalize each to 1 for simplicity in init
                U[:, i] /= norm_u
                V[:, i] /= norm_v
                W[:, i] /= norm_w
                # Scale one of them to carry the magnitude? 
                # Actually, let's just start with random small values.
                pass
        
        # Better init: Use schoolbook for first r terms?
        # If r < 84, we can't use all schoolbook terms.
        # Let's try random init with small variance.
        U = np.random.randn(dim_u, r) * 0.1
        V = np.random.randn(dim_v, r) * 0.1
        W = np.random.randn(dim_w, r) * 0.1
        
        prev_error = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            # Fix V, W
            # T_res = T - sum_{k != i} u_k v_k w_k
            # We want to find u_i to minimize || T_res - u_i v_i w_i ||^2
            # This is equivalent to:
            # (v_i \otimes w_i)^T (v_i \otimes w_i) u_i = (v_i \otimes w_i)^T T_res_flat
            # Let M_i = v_i \otimes w_i (vector of size dim_v * dim_w)
            # But we can update all U at once?
            # No, ALS usually updates one factor matrix at a time.
            
            # Update U:
            # For each column k, we solve for U[:, k].
            # The term involving U[:, k] is U[:, k] \otimes V[:, k] \otimes W[:, k].
            # Let's form the Khatri-Rao product of V and W: M_vw = V \odot W (size dim_v*dim_w x r)
            # The residual for the k-th term is T_flat - sum_{l != k} (U[:, l] \otimes M_vw[:, l])
            # Wait, the expansion is:
            # T_recon_flat = sum_k (U[:, k] \otimes (V[:, k] \odot W[:, k]))
            # So T_recon_flat = U_flat @ diag? No.
            # T_recon_flat = sum_k U[:, k] \otimes M_vw[:, k]
            # This can be written as a matrix multiplication if we unroll.
            # Let T_mat be a matrix of size (dim_u, dim_v*dim_w).
            # T_mat[u, idx] = T[u, v, w] where idx = v*dim_w + w.
            # Then T_recon_mat = sum_k U[:, k] * M_vw[:, k]^T ?
            # No.
            # T_recon[u, v, w] = sum_k U[u, k] V[v, k] W[w, k].
            # If we fix V and W, we want to find U to minimize || T - T_recon ||.
            # This is a linear least squares problem for each column of U?
            # No, for each row u of U, we want to find U[u, :] to minimize error.
            # Error squared = sum_{u,v,w} (T[u,v,w] - sum_k U[u,k] V[v,k] W[w,k])^2
            # Let's define a matrix A of size (dim_u, dim_v*dim_w) where A[u, v*dim_w+w] = V[v,k]W[w,k]?
            # No, V and W depend on k.
            
            # Standard ALS for CP:
            # Fix V, W.
            # For each u, we want to minimize || T[u, :, :] - sum_k U[u,k] (V[:,k] \otimes W[:,k]) ||^2
            # Let B_k = V[:,k] \otimes W[:,k] (vector of size dim_v*dim_w)
            # Let B be a matrix of size (dim_v*dim_w, r) with columns B_k.
            # Let T_u be the row u of T flattened (size dim_v*dim_w).
            # We want to find x = U[u, :] (size r) to minimize || T_u - x^T B ||^2
            # Solution: x = (B^T B)^{-1} B^T T_u
            # This is the same for all u!
            # So we can solve for the entire U matrix at once?
            # No, T_u is different for each u.
            # But the system matrix (B^T B) and the right-hand side structure is similar.
            # Let's compute M = B^T B (r x r) and C = B^T T_mat (r x dim_u)
            # Where T_mat is (dim_v*dim_w, dim_u) with T_mat[v*dim_w+w, u] = T[u, v, w].
            # Then U = M^{-1} C.
            
            # Let's construct T_mat
            # T is (dim_u, dim_v, dim_w)
            # T_mat = np.transpose(T, (1, 2, 0)).reshape(dim_v*dim_w, dim_u)
            
            # Compute B = V \odot W
            # B is (dim_v*dim_w, r)
            B = khatri_rao(V, W)
            
            # Compute M = B^T B
            M = B.T @ B
            # Add small regularization for stability
            M += 1e-9 * np.eye(r)
            
            # Compute C = B^T T_mat
            # T_mat is (dim_v*dim_w, dim_u)
            T_mat = np.transpose(T, (1, 2, 0)).reshape(dim_v*dim_w, dim_u)
            C = B.T @ T_mat
            
            # Solve M U^T = C  =>  U^T = M^{-1} C
            try:
                U_new = np.linalg.solve(M, C).T
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(M, C, rcond=None)[0].T
                
            U = U_new
            
            # Similarly for V
            # Fix U, W
            # B = U \odot W
            B = khatri_rao(U, W)
            M = B.T @ B
            M += 1e-9 * np.eye(r)
            # T_mat for V: (dim_u*dim_w, dim_v)
            T_mat_v = np.transpose(T, (0, 2, 1)).reshape(dim_u*dim_w, dim_v)
            C = B.T @ T_mat_v
            try:
                V_new = np.linalg.solve(M, C).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(M, C, rcond=None)[0].T
            V = V_new
            
            # Similarly for W
            # Fix U, V
            # B = U \odot V
            B = khatri_rao(U, V)
            M = B.T @ M
            M = B.T @ B
            M += 1e-9 * np.eye(r)
            # T_mat for W: (dim_u*dim_v, dim_w)
            T_mat_w = np.transpose(T, (0, 1, 2)).reshape(dim_u*dim_v, dim_w)
            C = B.T @ T_mat_w
            try:
                W_new = np.linalg.solve(M, C).T
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(M, C, rcond=None)[0].T
            W = W_new
            
            # Calculate error
            T_recon = np.einsum('uk,vk,wk->uvw', U, V, W)
            error = np.linalg.norm(T_recon - T)
            
            if error < tol:
                break
            if abs(prev_error - error) < 1e-12:
                break
            prev_error = error
            
        return U, V, W, error

    # Search for lower rank
    for r in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Try a few restarts
        num_restarts = 5 if r > 80 else 10
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            U, V, W, err = als_decompose(r, max_iters=30)
            if err < 1e-8:
                valid, res = verify(U, V, W)
                if valid:
                    best_result = res
                    best_rank = r
                    found = True
                    break
        
        if found:
            # If we found a rank r, we can stop or continue to find lower?
            # Usually, if we find one, we might try to find even lower.
            # But given time constraints, finding the first one is a win.
            # Let's try to go lower for a bit if we have time.
            continue
        
    return best_result
# EVOLVE-BLOCK-END
