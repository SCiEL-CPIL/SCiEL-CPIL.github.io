"""Research program: Attempts rank-14 construction via numerical optimization, falls back to schoolbook."""
import numpy as np
import scipy.optimize as so

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, k] = 1.0
    # Decomposition: sum_c U_c[i, j] * V_c[j, k] * W_c[k, i] = 1
    # U shape: (n, m) -> flattened (n*m)
    # V shape: (m, p) -> flattened (m*p)
    # W shape: (p, n) -> flattened (p*n)
    
    rank_candidate = 14
    
    # Reshape helper
    def flatten_tensors(params, rank, n, m, p):
        # params: 1D array of length rank * (n*m + m*p + p*n)
        U_flat = params[:rank * (n * m)].reshape(rank, n * m)
        V_flat = params[rank * (n * m):rank * (n * m + m * p)].reshape(rank, m * p)
        W_flat = params[rank * (n * m + m * p):].reshape(rank, p * n)
        return U_flat, V_flat, W_flat

    def objective(params):
        U_flat, V_flat, W_flat = flatten_tensors(params, rank_candidate, n, m, p)
        # Compute sum of products
        # We need sum_c U_c[i,j] V_c[j,k] W_c[k,i]
        # Reshape to (rank, n, m), (rank, m, p), (rank, p, n)
        U = U_flat.reshape(rank_candidate, n, m)
        V = V_flat.reshape(rank_candidate, m, p)
        W = W_flat.reshape(rank_candidate, p, n)
        
        # Sum over c
        # Result shape (n, p, n)? No.
        # For each i, j, k:
        # term = U[:, i, j] * V[:, j, k] * W[:, k, i]
        # Sum over c (axis 0)
        
        # Efficient way:
        # Loop over i, k?
        # Or use einsum
        # sum_c U_c[i,j] V_c[j,k] W_c[k,i]
        # This is not a standard contraction because indices cycle i->j->k->i.
        
        # Let's just loop since dimensions are small.
        error = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vector of products for each c
                    # U[:, i, j] is shape (rank,)
                    # V[:, j, k] is shape (rank,)
                    # W[:, k, i] is shape (rank,)
                    prods = U[:, i, j] * V[:, j, k] * W[:, k, i]
                    error[i, j, k] = np.sum(prods) - 1.0
        return error.flatten()

    # Try to find a solution
    # Initialize with random values
    np.random.seed(42)
    init_size = rank_candidate * (n * m + m * p + p * n)
    x0 = np.random.randn(init_size) * 0.1
    
    # Use least squares
    try:
        result = so.least_squares(objective, x0, method='lm', max_nfev=5000)
        if result.success or np.linalg.norm(result.fun) < 1e-5:
            U_flat, V_flat, W_flat = flatten_tensors(result.x, rank_candidate, n, m, p)
            
            # Round to nearest 0.5
            U_rounded = np.round(U_flat * 2) / 2
            V_rounded = np.round(V_flat * 2) / 2
            W_rounded = np.round(W_flat * 2) / 2
            
            # Verify exactness
            U = U_rounded.reshape(rank_candidate, n, m)
            V = V_rounded.reshape(rank_candidate, m, p)
            W = W_rounded.reshape(rank_candidate, p, n)
            
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prods = U[:, i, j] * V[:, j, k] * W[:, k, i]
                        val = np.sum(prods)
                        if not np.isclose(val, 1.0, atol=1e-9):
                            is_valid = False
                            break
                if not is_valid: break
            if not is_valid: break
            
            if is_valid:
                # Return in required format
                # U: (n*m, rank)
                # V: (m*p, rank)
                # W: (p*n, rank)
                U_out = U_rounded.T
                V_out = V_rounded.T
                W_out = W_rounded.T
                
                # Convert to list of lists of floats (or ints if integer)
                def to_list(mat):
                    return [[float(val) for val in row] for row in mat]
                
                return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
    except Exception:
        pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
