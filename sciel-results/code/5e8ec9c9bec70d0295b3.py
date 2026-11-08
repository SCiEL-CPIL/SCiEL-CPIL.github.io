import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    # T[I, J, K] where I=i*m+j, J=j*p+k, K=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Pre-compute slices for fast ALS
    # T_U[i] = T[i, :, :] shape (dim_v, dim_w)
    T_U = np.array([T[i, :, :] for i in range(dim_u)])
    # T_V[j] = T[:, j, :] shape (dim_u, dim_w)
    T_V = np.array([T[:, j, :] for j in range(dim_v)])
    # T_W[k] = T[:, :, k] shape (dim_u, dim_v)
    T_W = np.array([T[:, :, k] for k in range(dim_w)])

    def fast_als(U, V, W, max_iters=200, tol=1e-10):
        """Fast ALS with SVD-based updates."""
        for it in range(max_iters):
            # Update U: for each i, solve (V⊗W)^T U[i,:] = T[i,:,:].flatten()
            # M_U[c1,c2] = sum over j,k of V[j,c1]*W[k,c2]*T[i,j,k]
            # Actually: M_U = (V.T @ T_U[i] @ W) for each i
            # Better: M_U is r×r matrix where M_U[a,b] = sum_{j,k} V[j,a]*W[k,b]*T[i,j,k]
            # U_new[i,:] = M_U^{-1} @ (sum_{j,k} V[j,:]*W[k,:]*T[i,j,k])
            
            # Vectorized: for all i at once
            # G_U[i,a,b] = sum_{j,k} T[i,j,k] * V[j,a] * W[k,b]
            # b_U[i,a] = sum_{j,k} T[i,j,k] * V[j,a] * W[k,b]... wait
            # Actually we need: sum_c U[i,c] * (V[j,c]*W[k,c]) = T[i,j,k]
            # So: M_U[i] is r×r: M_U[i,a,b] = sum_{j,k} T[i,j,k]*V[j,a]*W[k,b]
            # And rhs[i,a] = sum_{j,k} T[i,j,k]*V[j,a]*W[k,b]... no
            # rhs[i,a] = sum_{j,k,b} T[i,j,k]*V[j,a]*W[k,b]... 
            # Let me think again.
            # sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k]
            # This is: (V.T @ diag(T[i,j,k]) @ W) ... no
            # For fixed i: we have a linear system in U[i,:]:
            # sum_c U[i,c] * (V[j,c] * W[k,c]) = T[i,j,k] for all j,k
            # Let M_U be r×r: M_U[a,b] = sum_{j,k} V[j,a]*W[k,b]*T[i,j,k]  -- NO
            # M_U should be: the coefficient matrix for U[i,:]
            # sum_c U[i,c] * V[j,c] * W[k,c] = T[i,j,k]
            # This is a system: for each (j,k), we have one equation
            # sum_c U[i,c] * (V[j,c]*W[k,c]) = T[i,j,k]
            # So the "design matrix" row for (j,k) is [V[j,0]*W[k,0], V[j,1]*W[k,1], ...]
            # Let D_U[j,k,c] = V[j,c]*W[k,c], shape (dim_v, dim_w, r)
            # Then: D_U.reshape(dim_v*dim_w, r) @ U[i,:] = T[i,:,:].flatten()
            # M_U = D_U_flat.T @ D_U_flat (r×r), rhs_U = D_U_flat.T @ T[i,:,:].flatten()
            # U[i,:] = M_U^{-1} @ rhs_U
            
            # Vectorized over i:
            D_U = np.einsum('jc,kc->jkc', V, W)  # (dim_v, dim_w, r)
            D_U_flat = D_U.reshape(dim_v * dim_w, r)
            M_U = D_U_flat.T @ D_U_flat  # (r, r)
            # rhs_U[i,a] = sum_{j,k,b} D_U_flat[j*dim_w+k, b] * T[i,j,k] * delta_{a,b}
            # = sum_{j,k} D_U_flat[j*dim_w+k, a] * T[i,j,k]
            # = sum_{j,k} V[j,a]*W[k,a] * T[i,j,k]
            rhs_U = np.einsum('jka,ijk->ia', D_U, T_U)  # (dim_u, r)
            
            try:
                U_new = np.linalg.solve(M_U, rhs_U.T).T
            except:
                U_new = np.linalg.lstsq(D_U_flat, T_U.reshape(dim_u, -1), rcond=None)[0].reshape(dim_u, r)
            
            # Update V
            D_V = np.einsum('ic,kc->ikc', U_new, W)  # (dim_u, dim_w, r)
            D_V_flat = D_V.reshape(dim_u * dim_w, r)
            M_V = D_V_flat.T @ D_V_flat
            rhs_V = np.einsum('ikc,ijk->jc', D_V, T_V)  # (dim_v, r)
            
            try:
                V_new = np.linalg.solve(M_V, rhs_V.T).T
            except:
                V_new = np.linalg.lstsq(D_V_flat, T_V.reshape(dim_v, -1), rcond=None)[0].reshape(dim_v, r)
            
            # Update W
            D_W = np.einsum('ic,jc->ijc', U_new, V_new)  # (dim_u, dim_v, r)
            D_W_flat = D_W.reshape(dim_u * dim_v, r)
            M_W = D_W_flat.T @ D_W_flat
            rhs_W = np.einsum('ijc,ijk->kc', D_W, T_W)  # (dim_w, r)
            
            try:
                W_new = np.linalg.solve(M_W, rhs_W.T).T
            except:
                W_new = np.linalg.lstsq(D_W_flat, T_W.reshape(dim_w, -1), rcond=None)[0].reshape(dim_w, r)
            
            # Check convergence
            diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
        
        return U, V, W

    def check_solution(U, V, W):
        """Check if solution is valid (close to half-integers and exact)."""
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        
        if (np.linalg.norm(U - U_h) > 0.05 or 
            np.linalg.norm(V - V_h) > 0.05 or 
            np.linalg.norm(W - W_h) > 0.05):
            return None
        
        T_approx = np.einsum('ic,jc,kc->ijk', U_h, V_h, W_h)
        if np.allclose(T_approx, T, atol=1e-8):
            return {
                "U": U_h.tolist(),
                "V": V_h.tolist(),
                "W": W_h.tolist()
            }
        return None

    def try_rank(r, max_restarts=100, deadline=None):
        """Try to find a rank-r decomposition using structured initializations."""
        best_result = None
        
        for restart in range(max_restarts):
            if deadline and time.time() > deadline:
                break
            
            np.random.seed(12345 + restart * 7)
            
            # Structured initialization: start near schoolbook and perturb
            if restart == 0:
                # Pure schoolbook initialization
                U = np.zeros((dim_u, r))
                V = np.zeros((dim_v, r))
                W = np.zeros((dim_w, r))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                            col += 1
                # Perturb
                U += np.random.randn(dim_u, r) * 0.1
                V += np.random.randn(dim_v, r) * 0.1
                W += np.random.randn(dim_w, r) * 0.1
            elif restart == 1:
                # Winograd-inspired: use (x+y) type combinations
                U = np.zeros((dim_u, r))
                V = np.zeros((dim_v, r))
                W = np.zeros((dim_w, r))
                # For 2x3, try to combine rows
                # Row 0: A[0,:], Row 1: A[1,:]
                # Col 0: B[:,0], Col 1: B[:,1], Col 2: B[:,2]
                # Winograd for 2x1x3: use (A[0]+A[1]) and (A[0]-A[1])
                # This gives rank reduction for the n=2 dimension
                idx = 0
                for j in range(m):
                    for k in range(p):
                        if idx < r:
                            # Use sum and difference of rows
                            U[0 * m + j, idx] = 1.0  # row 0
                            U[1 * m + j, idx] = 1.0  # row 1
                            V[j * p + k, idx] = 1.0
                            W[k * n + 0, idx] = 1.0
                            W[k * n + 1, idx] = 1.0
                        idx += 1
                # Add difference terms
                for j in range(m):
                    for k in range(p):
                        if idx < r:
                            U[0 * m + j, idx] = 1.0
                            U[1 * m + j, idx] = -1.0
                            V[j * p + k, idx] = 1.0
                            W[k * n + 0, idx] = 1.0
                            W[k * n + 1, idx] = -1.0
                        idx += 1
                # Fill remaining with random
                for c in range(idx, r):
                    U[:, c] = np.random.randn(dim_u) * 0.5
                    V[:, c] = np.random.randn(dim_v) * 0.5
                    W[:, c] = np.random.randn(dim_w) * 0.5
            else:
                # Random initialization with varying scale
                scale = 0.5 + 0.5 * np.random.rand()
                U = np.random.randn(dim_u, r) * scale
                V = np.random.randn(dim_v, r) * scale
                W = np.random.randn(dim_w, r) * scale
            
            # Run fast ALS
            U, V, W = fast_als(U, V, W, max_iters=100)
            
            # Check solution
            res = check_solution(U, V, W)
            if res:
                return res
        
        return None

    deadline = time.time() + 110  # Leave 40s margin for safety

    # Try rank 29 first
    res = try_rank(29, max_restarts=200, deadline=deadline)
    if res:
        return res
    
    # Try rank 28
    if time.time() < deadline:
        res = try_rank(28, max_restarts=100, deadline=deadline)
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
