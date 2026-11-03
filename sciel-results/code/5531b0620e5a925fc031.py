"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses structured search: start from schoolbook, try block-structured reductions,
    and local optimization with exact verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    def verify(U, V, W, tol=1e-10):
        """Verify decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Build tensor check efficiently
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                    # Actually need element-wise product then sum
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-10):
        """Faster verification using numpy"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by merging columns that have similar patterns
    # Strategy: for each pair of columns, check if they can be merged
    # Two columns c1, c2 can be merged if for all (i,j,k):
    #   U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] + U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
    # equals the original single contribution
    
    # More practical approach: try structured decompositions
    # For 4x4*4x9, try rank < 144 using block structure
    
    # Try: exploit the fact that for fixed j, we have n*p products
    # Group by j: for each j, we need to compute A[:,j] * B[j,:] which is outer product
    # This is a rank-1 operation per j, so total rank = m * (rank of outer product) = m * 1 = m? No...
    
    # Actually, the standard lower bound is max(n*p, m*p, n*m) = max(36, 36, 16) = 36
    # But that's for the bilinear complexity, not tensor rank in this formulation
    
    # Let me try a different structured approach:
    # Use the fact that we can combine terms
    
    # Try random search for lower rank
    np.random.seed(42)
    
    def try_random_rank(target_rank, n_trials=50):
        """Try to find a valid decomposition of given rank"""
        if time.time() > deadline - 5:
            return None
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            U_r = np.random.randn(n*m, target_rank) * 0.5
            V_r = np.random.randn(m*p, target_rank) * 0.5
            W_r = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple gradient-free check
            if verify_fast(U_r, V_r, W_r):
                return U_r, V_r, W_r
            # This is extremely unlikely to work randomly, so this is mainly for completeness
        
        return None
    
    # Try structured reduction: merge columns that only differ in one dimension
    # For schoolbook, column (i,j,k) has 1s at positions (i*m+j), (j*p+k), (k*n+i)
    # Two columns (i1,j1,k1) and (i2,j2,k2) can potentially be combined if...
    
    # Better approach: use the known result that for matrix multiplication,
    # the tensor rank is at least the bilinear complexity.
    # For 4x4*4x9, the best known upper bound is... let's just search.
    
    # Try a local search: start from schoolbook, randomly perturb and project
    # to maintain the constraint
    
    # Actually, let's try a more systematic approach:
    # The tensor T[i,j,k] = 1 for all (i,j,k).
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # This is equivalent to: for each (i,j,k), the "fiber" must sum to 1.
    
    # Key insight: if we can find vectors u_{i,j}, v_{j,k}, w_{k,i} in R^r such that
    # <u_{i,j}, v_{j,k} * w_{k,i}> = 1 (element-wise product then dot), then rank r works.
    
    # Let's try rank = 36 (which is n*p = m*p, the minimum dimension product)
    # Use: U[i*m+j, c] = a_{i,j,c}, V[j*p+k, c] = b_{j,k,c}, W[k*n+i, c] = c_{k,i,c}
    # such that sum_c a_{i,j,c} * b_{j,k,c} * c_{k,i,c} = 1
    
    # One structured approach: set a_{i,j,c} = delta_{i,c mod n} * something
    # This gets complicated. Let's just do a bounded search.
    
    # Try ranks from 143 down, but only a few values given time constraints
    # Focus on structured attempts for specific ranks
    
    # Attempt 1: Use block diagonal structure
    # For each j (0..3), the sub-problem is A[:,j] (4x1) * B[j,:] (1x9) = outer product (4x9)
    # The outer product of a 4-vector and 9-vector has tensor rank 1 in the bilinear sense
    # But in our formulation, for fixed j, we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,k
    # This is: for fixed j, the matrix M_j[i,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be all-ones
    # M_j = (U_j * V_j^T) element-wise times... no, it's a 3-tensor contraction
    
    # For fixed j, define U_j[i,c] = U[i*m+j,c], V_j[k,c] = V[j*p+k,c], W_j[k,i,c] = W[k*n+i,c]
    # Then sum_c U_j[i,c] * V_j[k,c] * W_j[k,i,c] = 1 for all i,k
    
    # If we set W_j[k,i,c] = 1 for all c, then we need sum_c U_j[i,c]*V_j[k,c] = 1 for all i,k
    # This means U_j * V_j^T = ones(4,9), which has rank 1 (outer product of ones vectors)
    # So U_j = ones(4,1) * alpha, V_j = ones(9,1) * (1/(9*alpha)) ... wait
    # U_j[i,c] * V_j[k,c] summed over c = 1 for all i,k
    # If we use rank 1: U_j[i,0] = a_i, V_j[k,0] = b_k, then a_i * b_k = 1 for all i,k
    # This requires a_i = 1/b_k for all i,k, so all a_i equal and all b_k equal
    # a * b = 1, so any a != 0, b = 1/a works.
    
    # So for each j, we can use rank 1! Total rank = m * 1 = 4?
    # Wait, but W is shared across j's. W[k*n+i, c] depends on k and i, not j.
    # So the same W columns are used for all j. This couples the j's.
    
    # Let me reconsider: for rank r, we need for ALL (i,j,k):
    # sum_{c=0}^{r-1} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # For fixed (i,k), varying j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
    # For fixed (j,k), varying i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
    
    # This is a system of n*m*p = 144 equations in r * (n*m + m*p + p*n) = r * (16 + 36 + 36) = 88r unknowns
    # For r = 2: 176 unknowns, 144 equations - might be feasible
    # For r = 1: 88 unknowns, 144 equations - underdetermined, likely no solution
    
    # Let's try to solve for small r using least squares + verification
    
    def solve_for_rank(target_rank, max_iters=100):
        """Try to find a decomposition of given rank using iterative optimization"""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(123)
        best_err = float('inf')
        best_solution = None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Simple iterative refinement: for each (i,j,k), adjust to reduce error
            # This is a nonlinear problem, so use simple gradient descent on the error
            # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            lr = 0.01
            for inner_it in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Compute errors
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i,j,k] = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :]) - 1.0
                
                err = np.sum(errors**2)
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        return U_r, V_r, W_r
                    best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                
                # Gradient w.r.t. U[i*m+j, c] = error[i,j,k] * V[j*p+k,c] * W[k*n+i,c] for each k
                # This is expensive, so do a few random coordinate updates
                for _ in range(20):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    c = np.random.randint(target_rank)
                    
                    # Gradient of error[i,j,k] w.r.t. U[i*m+j,c]
                    gU = errors[i,j,k] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    U_r[i*m+j, c] -= lr * gU
                    
                    gV = errors[i,j,k] * U_r[i*m+j, c] * W_r[k*n+i, c]
                    V_r[j*p+k, c] -= lr * gV
                    
                    gW = errors[i,j,k] * U_r[i*m+j, c] * V_r[j*p+k, c]
                    W_r[k*n+i, c] -= lr * gW
        
        if best_solution is not None and best_err < 1e-10:
            return best_solution
        return None
    
    # Try structured approach: use the block structure
    # For each j, we have a 4x9 outer product to compute
    # Use shared W structure
    
    # Attempt: rank = m * n * p / some_factor
    # Let's just try a few specific ranks with the solver
    
    # First, verify schoolbook works
    assert verify_fast(U, V, W), "Schoolbook should be valid"
    
    # Try to find lower rank
    # Given time constraints, try a few specific ranks
    candidate_ranks = [143, 140, 135, 130, 120, 100, 80, 60, 45, 36]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        result = solve_for_rank(target_rank, max_iters=20)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_fast(U_r, V_r, W_r):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
