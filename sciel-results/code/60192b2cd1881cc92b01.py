import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Build target tensor T[u_idx, v_idx, w_idx]
    # u_idx = i*m + j, v_idx = j'*p + k, w_idx = k*n + i
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_calc, T, atol=1e-9)

    def snap_and_verify(U, V, W):
        """Snap to half-integers and verify."""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_decompose(rank, max_iter=200, tol=1e-12):
        """Alternating least squares to find rank-rank decomposition."""
        np.random.seed(42)
        U = np.random.randn(U_size, rank)
        V = np.random.randn(V_size, rank)
        W = np.random.randn(W_size, rank)
        
        prev_cost = np.inf
        for it in range(max_iter):
            # Update U: for each u_idx, solve min ||T[u_idx,:,:] - V @ (W * U[u_idx,:])||
            # T_flat[u_idx, v_idx*W_size + w_idx] = sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
            # Let M_u = V @ W.T (V_size x W_size), then T[u_idx,:] = U[u_idx,:] @ M_u
            # Actually: T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
            # For fixed u: T[u,:,:] = sum_c U[u,c] * (V[:,c] * W[:,c])
            # Let N = V * W (element-wise, shape V_size x W_size... no, V_size x rank and W_size x rank)
            # N[v,w] = sum_c V[v,c] * W[w,c] = (V @ W.T)[v,w]
            N = V @ W.T  # (V_size, W_size)
            # T[u,:,:] = U[u,:] @ N  ->  T[u,:].reshape(1,-1) = U[u,:].reshape(1,-1) @ N
            # So: T_flat = U @ N where T_flat is (U_size, V_size*W_size)
            T_flat = T.reshape(U_size, V_size * W_size)
            N_flat = N.reshape(V_size * W_size)
            # Actually T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
            # This is NOT U @ (V @ W.T) in general.
            # Let me redo: for each c, the contribution to T is U[:,c] * V[:,c] * W[:,c] (outer product)
            # So T = sum_c outer(U[:,c], V[:,c], W[:,c])
            # For ALS on U: fix V,W. For each u_idx:
            #   T[u_idx, v_idx, w_idx] = sum_c U[u_idx,c] * (V[v_idx,c] * W[w_idx,c])
            #   Let M[u_idx, v_idx, w_idx] = V[v_idx,c] * W[w_idx,c] summed over c? No.
            #   M[v_idx, w_idx] = sum_c V[v_idx,c] * W[w_idx,c] = (V @ W.T)[v_idx, w_idx]
            #   Then T[u_idx, v_idx, w_idx] = U[u_idx, :] @ M[v_idx, w_idx]? No, that's not right either.
            #   
            # Correct ALS: T[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # Fix V,W. Define for each c: A_c = V[:,c] * W[:,c] (vector of length V_size*W_size)
            # Then T_flat[u, :] = sum_c U[u,c] * A_c
            # So T_flat = U @ A where A is (rank, V_size*W_size)
            A = np.zeros((rank, V_size * W_size))
            for c in range(rank):
                A[c, :] = np.outer(V[:, c], W[:, c]).flatten()
            # Solve: U_new = lstsq(A.T, T_flat.T).T
            U_new, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
            U_new = U_new.T
            
            # Update V: fix U,W. 
            # T[u,v,w] = V[v,c] * (U[u,c] * W[w,c])
            # For each v: T[:,v,:] = V[v,:] @ B where B is (rank, U_size*W_size)
            B = np.zeros((rank, U_size * W_size))
            for c in range(rank):
                B[c, :] = np.outer(U[:, c], W[:, c]).flatten()
            T_flat_v = T.transpose(1, 0, 2).reshape(V_size, U_size * W_size)
            V_new, _, _, _ = np.linalg.lstsq(B.T, T_flat_v.T, rcond=None)
            V_new = V_new.T
            
            # Update W: fix U,V.
            # T[u,v,w] = W[w,c] * (U[u,c] * V[v,c])
            # For each w: T[:,:,w] = W[w,:] @ C where C is (rank, U_size*V_size)
            C = np.zeros((rank, U_size * V_size))
            for c in range(rank):
                C[c, :] = np.outer(U[:, c], V[:, c]).flatten()
            T_flat_w = T.transpose(2, 0, 1).reshape(W_size, U_size * V_size)
            W_new, _, _, _ = np.linalg.lstsq(C.T, T_flat_w.T, rcond=None)
            W_new = W_new.T
            
            # Check convergence
            cost = np.sum((T - np.einsum('uc,vc,wc->uvw', U_new, V_new, W_new))**2)
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
            U, V, W = U_new, V_new, W_new
        
        return U, V, W

    # Try rank 10 first (known minimal)
    for rank in [10, 11, 12, 13, 14]:
        if time.time() > deadline:
            break
        
        max_restarts = 30 if rank == 10 else 15
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Different random seeds for diversity
            np.random.seed(1000 + restart * 7)
            
            try:
                U, V, W = als_decompose(rank, max_iter=300)
                if U is not None:
                    result = snap_and_verify(U, V, W)
                    if result:
                        U_f, V_f, W_f = result
                        return {
                            "U": U_f.tolist(),
                            "V": V_f.tolist(),
                            "W": W_f.tolist()
                        }
            except Exception:
                continue
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
