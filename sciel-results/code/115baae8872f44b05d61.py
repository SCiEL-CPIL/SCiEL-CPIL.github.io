import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin from 150s limit
    
    # Target tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 if (i, j, k) is valid, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Pre-compute flattened modes for ALS
    # Mode 0 (U): T reshaped to (n*m, m*p * p*n)
    T_flat_U = T.reshape(n*m, m*p * p*n)
    # Mode 1 (V): T transposed to (b, a, c) then reshaped to (m*p, n*m * p*n)
    T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    # Mode 2 (W): T transposed to (c, a, b) then reshaped to (p*n, n*m * m*p)
    T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    def compute_residual(U, V, W):
        """Compute the residual norm for a given decomposition."""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        M = np.einsum('ai,ci->aci', U, W)  # (n*m, p*n, rank)
        T_approx = np.einsum('aci,bi->abc', M, V)  # (n*m, m*p, p*n)
        R = T - T_approx
        return np.sum(R**2)
    
    def als_optimize(U, V, W, max_iters=100, tol=1e-10):
        """Alternating Least Squares optimization."""
        rank = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            K_U = np.einsum('bi,ci->bci', V, W).reshape(-1, rank)
            A_U = K_U.T @ K_U
            A_U += 1e-10 * np.eye(rank)
            B_U = T_flat_U @ K_U
            try:
                U = np.linalg.solve(A_U.T, B_U.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update V
            K_V = np.einsum('ai,ci->aci', U, W).reshape(-1, rank)
            A_V = K_V.T @ K_V
            A_V += 1e-10 * np.eye(rank)
            B_V = T_flat_V @ K_V
            try:
                V = np.linalg.solve(A_V.T, B_V.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update W
            K_W = np.einsum('ai,bi->abi', U, V).reshape(-1, rank)
            A_W = K_W.T @ K_W
            A_W += 1e-10 * np.eye(rank)
            B_W = T_flat_W @ K_W
            try:
                W = np.linalg.solve(A_W.T, B_W.T).T
            except np.linalg.LinAlgError:
                pass
            
            if it % 10 == 0 or it == max_iters - 1:
                residual = compute_residual(U, V, W)
                if residual < tol:
                    break
        
        return U, V, W, compute_residual(U, V, W)
    
    def try_round_and_verify(U, V, W):
        """Try to round to half-integers and verify."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        M_r = np.einsum('ai,ci->aci', U_rounded, W_rounded)
        T_approx_r = np.einsum('aci,bi->abc', M_r, V_rounded)
        if np.allclose(T_approx_r, T, atol=1e-9):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        return None
    
    # Schoolbook solution (rank 50)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try ranks from 49 down to 40
    for rank in range(rank_schoolbook - 1, max(35, rank_schoolbook - 10), -1):
        if time.time() > deadline - 20:
            break
        
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart * 1000 + rank)
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random initialization
                U = np.random.randn(n*m, rank) * 0.3
                V = np.random.randn(m*p, rank) * 0.3
                W = np.random.randn(p*n, rank) * 0.3
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.random.randn(n*m, rank) * 0.5
                V = np.random.randn(m*p, rank) * 0.5
                W = np.random.randn(p*n, rank) * 0.5
                mask = np.random.rand(n*m, rank) < 0.3
                U[mask] = 0
                mask = np.random.rand(m*p, rank) < 0.3
                V[mask] = 0
                mask = np.random.rand(p*n, rank) < 0.3
                W[mask] = 0
            else:
                # Structured initialization based on partial schoolbook
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for idx in range(min(rank, n*m*p)):
                    i = idx % n
                    j = (idx // n) % m
                    k = (idx // (n*m)) % p
                    if idx < rank:
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1
                U += np.random.randn(n*m, rank) * 0.1
                V += np.random.randn(m*p, rank) * 0.1
                W += np.random.randn(p*n, rank) * 0.1
            
            # Optimize using ALS
            U_opt, V_opt, W_opt, residual = als_optimize(U, V, W, max_iters=100)
            
            if residual < 1e-8:
                result = try_round_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
            
            # Also try a few gradient steps from the ALS result
            if residual < 1e-4 and time.time() < deadline - 10:
                # Try to refine with a few more iterations
                U_opt, V_opt, W_opt, residual = als_optimize(U_opt, V_opt, W_opt, max_iters=50)
                if residual < 1e-8:
                    result = try_round_and_verify(U_opt, V_opt, W_opt)
                    if result is not None:
                        return result
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
