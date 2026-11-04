import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _construct_via_linear_solve(n, m, p, rank, seed=42):
    """
    Attempts to find a rank decomposition by fixing U, V randomly and solving for W.
    Returns U, V, W if a valid low-residual solution is found, else None.
    """
    np.random.seed(seed)
    
    # Initialize U and V with random integers in {-1, 0, 1}
    # Using small integers helps in finding integer solutions later
    U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
    V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
    
    # Construct the linear system M * W_flat = b
    # Equation: sum_r U[i,j,r] * V[l,k,r] * W[k,i,r] = delta[j,l]
    # Indices:
    # i in [0, n), j in [0, m), k in [0, p), l in [0, m)
    # Unknowns: W[k, i, r] for k in [0, p), i in [0, n), r in [0, rank)
    # Let W_flat index be idx_w = (k * n + i) * rank + r
    
    num_eqs = n * m * p * m
    num_unk = p * n * rank
    
    M = np.zeros((num_eqs, num_unk))
    b = np.zeros(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(m):
                    # Target: delta[j,l]
                    b[eq_idx] = 1.0 if j == l else 0.0
                    
                    # Coefficients for W[k, i, r]
                    # Term: U[i,j,r] * V[l,k,r] * W[k,i,r]
                    for r in range(rank):
                        coeff = U[i * m + j, r] * V[l * p + k, r]
                        if coeff != 0:
                            w_idx = (k * n + i) * rank + r
                            M[eq_idx, w_idx] += coeff
                    eq_idx += 1
    
    # Solve least squares
    W_flat, residuals, rank_sol, sv = np.linalg.lstsq(M, b, rcond=None)
    
    # Check residual
    if len(residuals) > 0:
        res = residuals[0]
    else:
        res = np.linalg.norm(M @ W_flat - b)
    
    # If residual is very small, we have a solution
    if res < 1e-9:
        W = W_flat.reshape((p, n, rank))
        # Transpose W to shape (p*n, rank) with index k*n+i
        W_mat = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                W_mat[k * n + i, :] = W[k, i, :]
        
        # Round U, V, W to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W_mat = np.round(W_mat * 2) / 2
        
        # Verify exactness
        # Reconstruct tensor and check
        T_recon = np.zeros((n, m, p, m)) # i, j, k, l
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(m):
                        val = 0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[l * p + k, r] * W_mat[k * n + i, r]
                        T_recon[i, j, k, l] = val
        
        # Check if T_recon matches delta[j,l]
        # delta[j,l] is 1 if j==l else 0
        is_valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(m):
                        expected = 1.0 if j == l else 0.0
                        if abs(T_recon[i, j, k, l] - expected) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid: break
                if not is_valid: break
            if not is_valid: break
        
        if is_valid:
            return U, V, W_mat
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try increasing ranks
    # Schoolbook rank is n*m*p = 60
    # Strassen-like estimates might be lower.
    # Let's try a range of ranks.
    for rank in range(10, 61):
        for seed in range(50): # Try multiple seeds
            result = _construct_via_linear_solve(n, m, p, rank, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to schoolbook if no better found (should not happen for rank 60)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
