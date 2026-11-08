import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For matrix multiplication: T[i,j,k] = 1 if we're computing (AB)[i,k] using B's j-th column
    # Actually, the standard decomposition: (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k in the schoolbook
    
    # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    # In schoolbook: each (i,j,k) has exactly one term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # So the sum is 1. Correct.
    
    # Now try to reduce rank by finding linear dependencies
    # Strategy: try to express some columns as linear combinations of others
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a subset of columns that still spans the same tensor
    # Use greedy: try removing one column at a time if the tensor is still preserved
    
    # Build the full tensor from current decomposition
    def compute_tensor(U_mat, V_mat, W_mat):
        """Compute the tensor from factor matrices."""
        n_local, m_local, p_local = DIMENSIONS
        T = np.zeros((n_local, m_local, p_local))
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    # Sum over all columns
                    T[i, j, k] = np.sum(
                        U_mat[i * m_local + j, :] * V_mat[j * p_local + k, :] * W_mat[k * n_local + i, :]
                    )
        return T
    
    target_tensor = compute_tensor(U, V, W)
    
    # Greedy elimination: try to remove columns one by one
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank_school
    
    # Try to eliminate columns by checking if they're linearly dependent
    # For each column c, check if removing it still allows reconstruction
    # This is expensive, so we'll use a heuristic: try random subsets
    
    # Better approach: use the structure of matrix multiplication
    # For 3x5x6, we can use block decomposition
    
    # Actually, let's try a different strategy: use known constructions
    # For matrix multiplication, the rank can be reduced using Strassen-like methods
    
    # For 3x5x6, let's try to find a decomposition with rank around 30-50
    
    # Use a random search: start from schoolbook, randomly perturb and try to reduce
    
    # Simpler: try to find linear dependencies in the column space
    # The columns of U, V, W together define the decomposition
    # If a column is a linear combination of others in a compatible way, we can remove it
    
    # Let's try: for each pair of columns, check if one can be expressed in terms of the other
    # This is O(rank^2 * dimension) which is feasible for rank=90
    
    # Actually, the key insight: the tensor is fixed. We need min rank such that
    # there exist U, V, W with the tensor = sum_c u_c ⊗ v_c ⊗ w_c
    
    # This is a hard problem. Let's use a practical approach:
    # 1. Start with schoolbook
    # 2. Try to merge terms using algebraic identities
    
    # For matrix multiplication, we have:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # We can group terms by i and k, and for each pair, we need to represent
    # the linear form sum_j A[i,j] B[j,k]
    
    # For fixed i,k, this is a bilinear form in the j variables
    # The rank of this bilinear form is min(m, ...) but we need to share across i,k
    
    # Let's just do a local search: randomly try to replace 2 columns with 1
    
    # Time check
    if time.time() > deadline - 30:
        # Return schoolbook if no time for search
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to reduce by finding that some columns are redundant
    # For the schoolbook, columns are indexed by (i,j,k)
    # Column (i,j,k) has U nonzero at (i,j), V nonzero at (j,k), W nonzero at (k,i)
    
    # Two columns (i1,j1,k1) and (i2,j2,k2) can potentially be merged if...
    # Actually, they're independent in the schoolbook
    
    # Let's try a different approach: use the fact that for fixed j,
    # the terms involving A[:,j] and B[j,:] can be combined
    
    # For a fixed j, we have n*p terms: (i,j,k) for all i,k
    # These correspond to the outer product of column j of A and row j of B
    # The rank of this outer product is 1, so we need at least 1 term per j
    # But we can share terms across different j values
    
    # Minimum possible rank: at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # So rank 30 is a lower bound from the matrix dimensions
    
    # Let's aim for rank 30-40
    
    # Construct a rank-30 decomposition if possible
    # For rank 30, we need U: 15x30, V: 30x30, W: 18x30
    
    # One approach: use the fact that the tensor is a sum of rank-1 tensors
    # where each tensor corresponds to a "path" through the matrix multiplication
    
    # For now, let's try to find a good decomposition through optimization
    
    # Use scipy to solve for a reduced rank decomposition
    from scipy.optimize import minimize
    
    # Parameterize U, V, W with rank r
    def objective(params, r, n_local, m_local, p_local):
        U_flat = params[:n_local * m_local * r]
        V_flat = params[n_local * m_local * r : n_local * m_local * r + m_local * p_local * r]
        W_flat = params[n_local * m_local * r + m_local * p_local * r : ]
        
        U_mat = U_flat.reshape(n_local * m_local, r)
        V_mat = V_flat.reshape(m_local * p_local, r)
        W_mat = W_flat.reshape(p_local * n_local, r)
        
        # Compute tensor error
        error = 0.0
        for i in range(n_local):
            for j in range(m_local):
                for k in range(p_local):
                    val = np.sum(U_mat[i * m_local + j, :] * V_mat[j * p_local + k, :] * W_mat[k * n_local + i, :])
                    error += (val - 1.0) ** 2
        return error
    
    # Try different ranks
    for target_rank in [30, 35, 40, 45, 50]:
        if time.time() > deadline - 20:
            break
        
        # Initialize with schoolbook truncated
        init_U = U[:, :target_rank]
        init_V = V[:, :target_rank]
        init_W = W[:, :target_rank]
        
        # This won't work directly because truncating schoolbook doesn't give the right tensor
        # We need a proper initialization
        
        # Better: use random initialization
        np.random.seed(42)
        r = target_rank
        U_init = np.random.randn(n * m, r)
        V_init = np.random.randn(m * p, r)
        W_init = np.random.randn(p * n, r)
        
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            if restart > 0:
                U_init = np.random.randn(n * m, r)
                V_init = np.random.randn(m * p, r)
                W_init = np.random.randn(p * n, r)
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(objective, params, args=(r, n, m, p), method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-10})
            
            if result.fun < 1e-6:
                # Found a solution, round to integers/half-integers
                U_sol = result.x[:n * m * r].reshape(n * m, r)
                V_sol = result.x[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
                W_sol = result.x[n * m * r + m * p * r :].reshape(p * n, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
                if np.max(np.abs(T_check - target_tensor)) < 1e-6:
                    best_rank = r
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    found = True
                    break
        
        if found:
            break
    
    # If no reduced rank found, return schoolbook
    if best_rank == rank_school:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
