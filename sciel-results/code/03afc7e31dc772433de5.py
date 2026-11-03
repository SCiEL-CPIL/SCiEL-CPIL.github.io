"""Tensor decomposition for 3x3 by 3x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Use tensor contraction method: for a given rank r, fix U and V,
    # then solve linear system for W to see if T is in the span.
    
    # Start with known good ranks and work down
    # For 3x3 * 3x5, schoolbook is 45. Let's try to find lower rank.
    
    def try_rank(r, seed=42):
        """Try to find a rank-r decomposition using tensor contraction."""
        rng = np.random.RandomState(seed)
        
        # Initialize U and V randomly
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        
        # Build the linear system: for each (i,j,k), we need
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is linear in W entries.
        
        # Number of equations: n*m*p = 45
        # Number of unknowns: p*n*r = 15*r
        # We need r >= ceil(45/15) = 3 for a square/overdetermined system
        # But actually we need the system to be consistent.
        
        # Build matrix A where A[(i,j,k), (c, w_idx)] = U[i*m+j, c] * V[j*p+k, c]
        # where w_idx = k*n + i
        
        num_eq = n * m * p
        num_unk = p * n * r
        
        # A is num_eq x num_unk
        # For equation (i,j,k) and unknown (c, w_idx):
        # Coefficient = U[i*m+j, c] * V[j*p+k, c] if w_idx == k*n+i, else 0
        
        # More efficiently: for each c, the contribution to W[:, c] is:
        # W[:, c] should satisfy: for all (i,j,k),
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = part of sum
        # Actually, let's build it properly.
        
        # Let's reshape: for each c, define a tensor A_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c]
        # Then we need sum_c A_c[i,j,k] * W[k*n+i, c] = 1
        
        # This is a linear system in the variables W[k*n+i, c] for all k,i,c.
        
        # Build the matrix
        A = np.zeros((num_eq, num_unk), dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for c in range(r):
                        # W index for this c is at position c * (p*n) + (k*n + i)
                        w_idx = c * (p * n) + (k * n + i)
                        A[eq_idx, w_idx] = U[i * m + j, c] * V[j * p + k, c]
        
        b = np.ones(num_eq, dtype=np.float64)
        
        # Solve the linear system
        try:
            # Use least squares if overdetermined
            if num_unk >= num_eq:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is valid
                if rank_sol < num_unk:
                    # Rank deficient, might not work well
                    pass
            else:
                # Underdetermined, use lstsq
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Verify the solution
            W_mat = W_flat.reshape(r, p * n).T  # shape (p*n, r)
            
            # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W_mat[k*n+i, c] for c in range(r))
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                return U, V, W_mat, max_err
            else:
                return None
        except np.linalg.LinAlgError:
            return None
    
    # Try ranks from low to high
    # Minimum possible rank: need at least ceil(n*m*p / (p*n)) = ceil(m) = 3
    # But realistically, for matrix mult, we expect much higher.
    # Let's try a range of ranks with different seeds
    
    best_result = None
    best_rank = None
    best_err = float('inf')
    
    for r in range(3, 50):
        for seed in range(20):
            result = try_rank(r, seed)
            if result is not None:
                U, V, W_mat, err = result
                if err < best_err:
                    best_err = err
                    best_rank = r
                    best_result = (U, V, W_mat)
                    if r <= 30:  # Found a good rank, stop early
                        break
        if best_rank is not None and best_rank <= 30:
            break
    
    if best_result is not None:
        U, V, W_mat = best_result
        U_list = [[float(U[i, j]) for j in range(best_rank)] for i in range(n * m)]
        V_list = [[float(V[i, j]) for j in range(best_rank)] for i in range(m * p)]
        W_list = [[float(W_mat[i, j]) for j in range(best_rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        
        U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
        V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
        W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
