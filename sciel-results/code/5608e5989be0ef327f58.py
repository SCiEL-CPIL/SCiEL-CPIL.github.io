"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Target rank: try to find a decomposition with rank < 36
    # Lower bound is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    
    # Build the target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[i*m + j, j'*p + k, k'*n + i'] = 1 if j==j', k==k', i==i' else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try ranks from 18 upward
    for rank in range(18, 37):
        # Initialize with random values
        np.random.seed(rank)  # Reproducible
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Alternating minimization
        for iteration in range(5000):
            # Update U: for each column r, minimize ||T - sum_{r'} u_{r'} ⊗ v_{r'} ⊗ w_{r'}||^2
            # This is a linear system for each r
            # Fix V and W, solve for U
            # T_flat = sum_r U[:,r] * (V[:,r] ⊗ W[:,r])^T
            # So for each r, U[:,r] = argmin ||T_flat - sum_{r'≠r} U[:,r']*(V[:,r']⊗W[:,r'])^T - U[:,r]*(V[:,r]⊗W[:,r])^T||^2
            # This is: U[:,r] = (V[:,r]⊗W[:,r])^T T_flat / ||V[:,r]⊗W[:,r]||^2 (if we ignore other terms)
            # Better: use least squares
            
            # Flatten T
            T_flat = T.reshape(-1)
            
            # Compute residual without current column
            # For efficiency, maintain a running sum
            # Let's do proper alternating least squares
            
            # Precompute V⊗W for each r
            VW = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                VW[:, r] = (V[:, r] * W[:, r]).reshape(-1)  # This is outer product flattened
            
            # Now T_flat ≈ VW @ U_flat where U_flat is (rank, n*m) reshaped
            # Actually: T_flat = sum_r VW[:,r] * U_flat[r,:] where U_flat[r,:] = U[:,r]
            # So T_flat = VW @ U_flat^T ... let me be careful with shapes
            
            # U has shape (n*m, rank), so U_flat[r, i] = U[i, r]
            # T_flat[a] = sum_r VW[a, r] * U[i_a, r] where a corresponds to (i_a, b_a, c_a)
            # So T_flat = sum_r VW[:,r] * U[:,r]^T ... no
            
            # Let me define: for each r, the contribution is np.outer(V[:,r] * W[:,r], U[:,r])? No.
            # The tensor is sum_r u_r ⊗ v_r ⊗ w_r
            # Flattened: for index (a,b,c), T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # So T_flat = sum_r np.kron(np.kron(U[:,r], V[:,r]), W[:,r])? No, that's not right.
            # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Flattened in order (a,b,c): T_flat = sum_r np.einsum('ar,br,cr->abc', U, V, W).reshape(-1)
            
            # For ALS, fixing V and W, we solve for U:
            # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let M_r[b,c] = V[b,r] * W[c,r]
            # Then T[:,b,c] = sum_r U[:,r] * M_r[b,c]
            # For fixed b,c: T[:,b,c] = U @ (V[:,b] * W[:,c]) ... no
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b,c: T[:,b,c] = U @ (V[:,b] * W[:,c]) where V[:,b] is a vector of length rank with entries V[b,r], similarly for W
            # So: T[:,b,c] = U @ (V[b,:] * W[c,:])
            # Let's define for each (b,c): d_{b,c} = V[b,:] * W[c,:] (element-wise product, length rank)
            # Then T[:,b,c] = U @ d_{b,c}
            # So for all (b,c): T_reshaped = U @ D where D has shape (rank, m*p * p*n) with D[r, b*p*n+c] = V[b,r] * W[c,r]
            
            # Similarly for V and W.
            
            # Let's just do it directly with einsum for clarity, even if slower
            # Given time constraints, let's use a simpler approach
            
            # Compute current residual
            current = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T - current
            if np.linalg.norm(residual) < 1e-6:
                break
            
            # Update U
            # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # Let's define for each r: M_r = V[:,r] ⊗ W[:,r] (shape m*p, p*n)
            # T[a,:,:] = sum_r U[a,r] * M_r
            # So T_reshaped (n*m, m*p*p*n) = U @ M_reshaped (rank, m*p*p*n)
            M = np.zeros((rank, m*p * p*n))
            for r in range(rank):
                M[r, :] = (V[:, r] * W[:, r]).reshape(-1)
            T_reshaped = T.reshape(n*m, m*p * p*n)
            # Solve U @ M^T = T_reshaped^T => U = T_reshaped^T @ (M @ M^T)^{-1} M^T ... 
            # Least squares: U = (M @ M^T)^{-1} M @ T_reshaped^T ... 
            # Actually: T_reshaped = U @ M^T where U is (n*m, rank), M is (rank, m*p*p*n)
            # So U = T_reshaped @ (M^T @ M)^{-1} M^T ... no
            # T_reshaped (n*m, K) = U (n*m, rank) @ M^T (rank, K) where K = m*p*p*n
            # So U = T_reshaped @ pinv(M^T)
            U_new = T_reshaped @ np.linalg.pinv(M.T)
            U = U_new
            
            # Update V
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b: T[:,:,b] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
            # T_reshaped2 (m*p, n*m * p*n) = V (m*p, rank) @ N^T (rank, n*m*p*n)
            N = np.zeros((rank, n*m * p*n))
            for r in range(rank):
                N[r, :] = (U[:, r] * W[:, r]).reshape(-1)
            T_reshaped2 = T.reshape(m*p, n*m * p*n)
            V = T_reshaped2 @ np.linalg.pinv(N.T)
            
            # Update W
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
            # T_reshaped3 (p*n, n*m * m*p) = W (p*n, rank) @ P^T (rank, n*m*m*p)
            P = np.zeros((rank, n*m * m*p))
            for r in range(rank):
                P[r, :] = (U[:, r] * V[:, r]).reshape(-1)
            T_reshaped3 = T.reshape(p*n, n*m * m*p)
            W = T_reshaped3 @ np.linalg.pinv(P.T)
        
        # Check if we found a good solution
        current = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T - current)
        if residual < 1e-4:
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            current_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            residual_rounded = np.linalg.norm(T - current_rounded)
            
            if residual_rounded < 1e-6:
                # Convert to lists
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
