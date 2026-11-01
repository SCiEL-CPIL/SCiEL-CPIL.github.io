import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
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
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def als_decompose(rank, max_iters=100, tol=1e-8):
        """
        Alternating Least Squares for tensor decomposition.
        """
        # Random initialization
        # Use small values to start
        U = np.random.randn(U_size, rank) * 0.1
        V = np.random.randn(V_size, rank) * 0.1
        W = np.random.randn(W_size, rank) * 0.1
        
        # Precompute flattened T
        T_flat = T.flatten()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            prev_U = U.copy()
            
            # Update U
            # Matrix M_U of shape (U_size * rank, V_size * W_size)
            # M_U[(u, r), (v, w)] = V[v, r] * W[w, r]
            # We want to solve M_U * U_flat = T_flat
            # U_flat is of shape (U_size * rank)
            
            # Efficient construction of M_U
            # V is (V_size, rank), W is (W_size, rank)
            # We can use broadcasting
            # V[:, :, None] * W[None, :, :] -> (V_size, W_size, rank)
            # Reshape to (V_size * W_size, rank)
            # Then we need to map (u, r) to rows.
            # This is a block diagonal structure if we group by r?
            # Actually, for each r, the row block for U[:, r] corresponds to the column r of V and W.
            # Let's just iterate over r for clarity, it's small.
            
            U_new = np.zeros_like(U)
            V_new = np.zeros_like(V)
            W_new = np.zeros_like(W)
            
            # Precompute matrices for updates
            # For U update:
            # For each r, we need A_r = outer(V[:, r], W[:, r]).flatten()
            # Then T_flat can be seen as sum_r U[:, r] \otimes A_r
            # Actually, T_flat[(u, v, w)] = sum_r U[u, r] V[v, r] W[w, r]
            
            # Let's construct the big linear system for U
            # Vectorize T: index = u * V_size * W_size + v * W_size + w
            # Vectorize U: index = u * rank + r
            # This doesn't map directly to a single dense matrix easily without heavy sparsity handling or Kronecker products.
            
            # Simpler: Iterate over r is not independent.
            # Standard ALS:
            # Fix V, W.
            # For each u, T[u, :, :] = sum_r U[u, r] (V[:, r] \otimes W[:, r])
            # Let A_r = V[:, r] \otimes W[:, r] (size V_size * W_size)
            # Let B_r = A_r
            # We have T_u (size V_size * W_size) = sum_r U[u, r] B_r
            # This is a linear system for U[u, :] given T[u, :].
            # But U[u, r] appears in the equation for T[u, :].
            # It does NOT appear in T[u', :] for u' != u.
            # So we can solve for each row u independently!
            
            # Matrix B of shape (V_size * W_size, rank)
            # B[:, r] = V[:, r] \otimes W[:, r]
            B = np.zeros((V_size * W_size, rank))
            for r in range(rank):
                B[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # For each u:
            # T[u, :, :].flatten() = B.T @ U[u, :]
            # U[u, :] = (B @ B.T)^-1 B @ T[u, :, :].flatten()
            # This is a rank x rank system for each u.
            
            BBT_inv = np.linalg.pinv(B @ B.T) # rank x rank
            
            for u in range(U_size):
                target_u = T[u, :, :].flatten()
                U_new[u, :] = BBT_inv @ (B.T @ target_u)
            
            U = U_new
            
            # Update V
            # Fix U, W.
            # For each v: T[:, v, :].flatten() = C.T @ V[v, :]
            # C[:, r] = U[:, r] \otimes W[:, r]
            C = np.zeros((U_size * W_size, rank))
            for r in range(rank):
                C[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            CCT_inv = np.linalg.pinv(C @ C.T)
            
            for v in range(V_size):
                target_v = T[:, v, :].flatten()
                V_new[v, :] = CCT_inv @ (C.T @ target_v)
            
            V = V_new
            
            # Update W
            # Fix U, V.
            # For each w: T[:, :, w].flatten() = D.T @ W[w, :]
            # D[:, r] = U[:, r] \otimes V[:, r]
            D = np.zeros((U_size * V_size, rank))
            for r in range(rank):
                D[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            DDT_inv = np.linalg.pinv(D @ D.T)
            
            for w in range(W_size):
                target_w = T[:, :, w].flatten()
                W_new[w, :] = DDT_inv @ (D.T @ target_w)
            
            W = W_new
            
            # Check convergence
            if np.allclose(U, prev_U, atol=tol):
                break
        
        return U, V, W

    # Try ranks from 14 upwards
    for rank in [14, 15, 16]:
        if time.time() > deadline:
            break
        
        max_restarts = 50 if rank == 14 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 1000 + rank)
            
            U_init, V_init, W_init = als_decompose(rank, max_iters=200)
            
            result = snap_and_verify(U_init, V_init, W_init)
            if result:
                U_final, V_final, W_final = result
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        
        # If ALS didn't work, try a few random restarts with least_squares on the full tensor?
        # ALS is usually sufficient for this size.

    # Fallback to schoolbook
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
