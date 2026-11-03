import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    # The matmul tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Try ranks from a lower bound upward, using random init + ALS
    # Theoretical lower bound: max(rank of flattening) 
    # Flattening (i,j) x k: 6x5 all ones -> rank 1
    # Flattening i x (j,k): 2x15 -> rank 2  
    # Flattening (i,k) x j: 10x3 -> rank 3
    # So lower bound is 3, but actual rank is likely higher
    
    # Let's try to find a decomposition with rank < 30
    # Use random initialization and alternating optimization
    
    def try_rank(r, seed=42):
        rng = np.random.default_rng(seed)
        
        # Initialize factors
        U = rng.standard_normal((U_dim, r))
        V = rng.standard_normal((V_dim, r))
        W = rng.standard_normal((W_dim, r))
        
        # Precompute indices for the 30 tensor entries
        tensor_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_indices.append((i * m + j, j * p + k, k * n + i))
        
        # Alternating least squares
        for iteration in range(2000):
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j, :] @ (V[j*p+k,:] * W[k*n+i,:])_c = 1 for each k
            # Let M_k[c] = V[j*p+k,c] * W[k*n+i,c]
            # Then U[i*m+j,:] @ M_k = 1 for all k
            # Stack: U[i*m+j,:] @ M_all = ones(p)
            # where M_all is p x r matrix
            
            for i in range(n):
                for j in range(m):
                    # Build M: p x r matrix where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j, :] @ M = ones(p)
                    # This is underdetermined if r > p, overdetermined if r < p
                    # Use least squares
                    U[i * m + j, :], residuals, rank_u, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
            # Let N_i[c] = U[i*m+j,c] * W[k*n+i,c]
            # V[j*p+k,:] @ N_i = 1 for all i
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, r))
                    for i in range(n):
                        N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    V[j * p + k, :], residuals, rank_v, sv = np.linalg.lstsq(N.T, np.ones(n), rcond=None)
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # Let P_j[c] = U[i*m+j,c] * V[j*p+k,c]
            # W[k*n+i,:] @ P_j = 1 for all j
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, r))
                    for j in range(m):
                        P[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    W[k * n + i, :], residuals, rank_w, sv = np.linalg.lstsq(P.T, np.ones(m), rcond=None)
        
        # Check if we found a valid decomposition
        # Compute the reconstructed tensor and check against ones
        max_err = 0.0
        for (ui, vi, wi) in tensor_indices:
            val = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
            max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        return None, None, None, max_err
    
    # Try decreasing ranks from 25 down to 10
    # Start with reasonable guesses
    for r in range(25, 9, -1):
        for seed in range(50):
            U, V, W, err = try_rank(r, seed=seed)
            if U is not None and err < 1e-6:
                # Convert to Python lists
                U_list = [[float(U[i, c]) for c in range(r)] for i in range(U_dim)]
                V_list = [[float(V[i, c]) for c in range(r)] for i in range(V_dim)]
                W_list = [[float(W[i, c]) for c in range(r)] for i in range(W_dim)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
