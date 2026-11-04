import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, with safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # W[k*n + i, r] convention
    
    # Build schoolbook decomposition
    rank_school = n * m * p
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
    
    # Verify schoolbook
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i,j,k are valid indices (i.e., the product term exists)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # Now try to reduce rank
    # Approach: find the minimum rank by checking linear dependencies
    
    # The rank of the tensor can be found by:
    # 1. Flattening the tensor in different ways
    # 2. The rank is at least the maximum of the flattening ranks
    # 3. But the tensor rank can be higher than any flattening rank
    
    # For our purposes, let's try a random search approach:
    # Start with rank r = rank_school, try to reduce by finding linear combinations
    
    # Better approach: use the fact that if we have a rank-r decomposition,
    # we can try to find a rank-(r-1) decomposition by checking if the
    # (r-1)-th secant variety contains our tensor.
    
    # Practical approach: 
    # 1. Try to find linear dependencies among the rank-1 terms
    # 2. Use iterative rank reduction
    
    # Let me try a different approach: random linear combinations
    # If we have columns u_c, v_c, w_c for c=0..r-1, we can try
    # new combinations and see if we can eliminate some columns.
    
    # Actually, the most effective approach for small tensors:
    # Enumerate possible rank values from a lower bound up to schoolbook,
    # and for each rank r, try to find a decomposition.
    
    # Lower bound: max of flattening ranks
    # Flattening 1: (n*m) x (m*p) matrix, rank = ?
    # The tensor T[i,j,k] = delta (it's 1 for all i,j,k in range)
    # Wait no. The matrix multiplication tensor T has T[i,j,k] = 1 for all valid indices.
    # So the (n*m) x (m*p) flattening is all-ones, which has rank 1.
    # That's not helpful.
    
    # Let me reconsider. The tensor for matrix multiplication:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The flattening along (i,j) vs k: matrix of size (n*m) x p, all ones, rank 1
    # The flattening along (i,k) vs j: matrix of size (n*p) x m, all ones, rank 1  
    # The flattening along (j,k) vs i: matrix of size (m*p) x n, all ones, rank 1
    
    # So all flattening ranks are 1, giving lower bound 1. Not helpful.
    
    # The actual rank is determined by the bilinear structure.
    
    # Let me just try the random reduction approach:
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies
    # Key idea: if we have a rank-r decomposition, and we can find
    # coefficients alpha_c such that sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0,
    # then we can eliminate one term.
    
    # More practically: try random linear combinations of columns
    # and check if the resulting decomposition is still valid.
    
    # A decomposition is valid if for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's define the tensor values
    # T[i,j,k] = 1 for all valid i,j,k
    
    def check_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Check if U,V,W form a valid decomposition of the matmul tensor."""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank reduction from schoolbook down
    # For each target rank r, try to find a decomposition
    
    # Strategy: start with schoolbook, try to eliminate columns one at a time
    # by expressing them as linear combinations of others.
    
    # If column c can be written as a linear combination of other columns
    # in a way that preserves the tensor identity, we can eliminate it.
    
    # This is equivalent to finding if the tensor lies in the span of
    # rank-1 tensors excluding one.
    
    # Practical approach: iterative elimination with random restarts
    import random
    random.seed(42)
    
    current_rank = rank_school
    cur_U = U.copy()
    cur_V = V.copy()
    cur_W = W.copy()
    
    # Try to eliminate columns
    max_iterations = 2000
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
            
        if current_rank <= 1:
            break
            
        # Pick a random column to try to eliminate
        c = random.randint(0, current_rank - 1)
        
        # Try to express column c as a linear combination of other columns
        # such that the tensor identity is preserved.
        
        # The condition is: for all i,j,k:
        # sum_{r != c} alpha_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] 
        #   = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # This is a system of linear equations in alpha_r (for r != c)
        # Number of equations: n*m*p = 56
        # Number of unknowns: current_rank - 1
        
        # If current_rank - 1 >= 56 and the system is consistent, we can eliminate.
        
        if current_rank - 1 < n * m * p:
            # Not enough unknowns to satisfy all equations generally
            # But maybe the system is still consistent due to structure
            pass
        
        # Build the system
        # For each (i,j,k), we need:
        # sum_{r != c} alpha_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # This is linear in alpha_r
        # Let's build the matrix
        
        if current_rank - 1 < n * m * p:
            # Too few unknowns for a general solution, skip
            continue
        
        # Actually, let's always try
        A_sys = np.zeros((n*m*p, current_rank - 1))
        b_sys = np.zeros(n*m*p)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs = cur_U[i*m+j, c] * cur_V[j*p+k, c] * cur_W[k*n+i, c]
                    b_sys[eq] = rhs
                    for r in range(current_rank):
                        if r == c:
                            continue
                        idx = r if r < c else r - 1
                        A_sys[eq, idx] = cur_U[i*m+j, r] * cur_V[j*p+k, r] * cur_W[k*n+i, r]
                    eq += 1
        
        # Solve A_sys @ alpha = b_sys
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            # Check if solution is valid
            if np.allclose(A_sys @ alpha, b_sys, atol=1e-9):
                # Eliminate column c
                # Remove column c from U, V, W
                mask = [r for r in range(current_rank) if r != c]
                cur_U = cur_U[:, mask]
                cur_V = cur_V[:, mask]
                cur_W = cur_W[:, mask]
                current_rank -= 1
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = cur_U.copy()
                    best_V = cur_V.copy()
                    best_W = cur_W.copy()
        except:
            continue
    
    # Convert to lists
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
