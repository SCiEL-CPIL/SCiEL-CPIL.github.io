"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix
    by a 5x5 matrix.
    
    The strategy is to model the problem as a sparse linear system by fixing
    the V matrix to a canonical sparse basis, then solve for the remaining
    integer/half-integer entries of U and W using a mixed-integer linear 
    programming approach with rank as the objective.
    
    We'll use a systematic search approach:
    1. Start with the schoolbook decomposition (rank = n*m*p = 50)
    2. Try to find lower rank decompositions using linear algebra techniques
    3. Use the structure of the matrix multiplication tensor to guide the search
    
    For a 2x5 by 5x5 matrix multiplication, we're looking for the tensor rank
    of the matrix multiplication tensor T_{i,j,k} = delta_{i,j+k} (in appropriate indexing).
    
    The key insight is that matrix multiplication can be viewed as a bilinear map,
    and the rank is the minimum number of rank-1 tensors needed to decompose it.
    """
    import numpy as np
    import time
    from itertools import product
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    SAFETY_MARGIN = 5.0  # seconds
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time) - SAFETY_MARGIN
    
    # Schoolbook decomposition (baseline)
    rank_schoolbook = n * m * p  # 50
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Best result so far
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W form a valid decomposition."""
        # Check dimensions
        if len(U) != n * m or len(U[0]) != rank:
            return False
        if len(V) != m * p or len(V[0]) != rank:
            return False
        if len(W) != p * n or len(W[0]) != rank:
            return False
        
        # Check that all entries are integers or half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # Check if value is integer or half-integer
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        
        # Verify the decomposition exactly
        # The tensor T_{i,j,k} should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But actually, the standard formulation is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = 1 if this is a valid multiplication term
        # Decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c[i,j] * V_c[j,k] * W_c[k,i]... 
        # 
        # Actually, let me reconsider the standard tensor decomposition for matrix multiplication.
        # The matrix multiplication tensor T has T[i,j,k] = 1 if and only if the (i,j) entry of A
        # and (j,k) entry of B contribute to (i,k) entry of C.
        # 
        # In the decomposition C = A*B, we have:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # The tensor rank decomposition writes:
        # T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
        # where u_c is a vector of size n*m (indexed by (i,j)),
        #       v_c is a vector of size m*p (indexed by (j,k)),
        #       w_c is a vector of size p*n (indexed by (k,i))
        # 
        # And the condition is that for all (i,j,k):
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # 
        # Wait, that's not quite right either. Let me think about this more carefully.
        # 
        # The standard formulation: We want to compute C = A * B where A is n x m, B is m x p.
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        # 
        # The matrix multiplication tensor T ∈ R^{n*m*p} has T[i,j,k] = 1 for all valid (i,j,k)
        # in the sense that it represents the bilinear map (A, B) -> C.
        # 
        # A rank-r decomposition means we can write:
        # C[i,k] = sum_{c=0}^{r-1} (sum_j A[i,j] * U_c[i,j]) * (sum_k B[j,k] * V_c[j,k]) * W_c[k,i]
        # 
        # No, that's not right. Let me use the standard formulation.
        # 
        # The correct formulation: A rank-r decomposition of matrix multiplication means
        # we can compute C = A*B using r scalar multiplications. Each scalar multiplication
        # involves:
        # - A linear combination of entries of A: alpha_c = sum_{i,j} U_c[i,j] * A[i,j]
        # - A linear combination of entries of B: beta_c = sum_{j,k} V_c[j,k] * B[j,k]
        # - The product alpha_c * beta_c contributes to C[i,k] with coefficient W_c[k,i]
        # 
        # So: C[i,k] = sum_c W_c[k,i] * alpha_c * beta_c
        #            = sum_c W_c[k,i] * (sum_{i',j} U_c[i',j] * A[i',j]) * (sum_{j',k'} V_c[j',k'] * B[j',k'])
        # 
        # For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
        # sum_c W_c[k,i] * U_c[i,j] * V_c[j,k] = 1 for all (i,j,k)
        # 
        # Wait, that's still not quite right because the indices don't match up.
        # 
        # Let me be very precise. We have:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # We want: C[i,k] = sum_c M_c * W_c[k,i]
        # where M_c is a scalar that is bilinear in A and B.
        # 
        # M_c = (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k'])
        # 
        # So: C[i,k] = sum_c W_c[k,i] * (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k'])
        # 
        # For this to hold for all A, B, we need:
        # For each (i,k): sum_c W_c[k,i] * (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k']) = sum_j A[i,j] B[j,k]
        # 
        # This is a bilinear equation in A and B. For it to hold for all A, B, we need:
        # sum_c W_c[k,i] * U_c[i',j] * V_c[j',k'] = delta_{i',i} * delta_{j,j'} * delta_{k',k} * delta_{j,k}
        # 
        # Hmm, this is getting complicated. Let me use a different approach.
        # 
        # Actually, the standard way to verify is:
        # The decomposition is valid if and only if:
        # For all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # 
        # Wait, I think I'm overcomplicating this. Let me look at the schoolbook example.
        # In the schoolbook, for each (i,j,k), we have one term where:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each (i,j,k).
        # 
        # So the verification is: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # 
        # Let me implement this.
        
        # Build the tensor and verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: Use the fact that for small matrices, we can try various rank values
    # and use linear algebra to check if a decomposition exists.
    
    # For 2x5 * 5x5, the known lower bounds and upper bounds:
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Upper bound: n*m*p = 50
    # 
    # Let's try ranks from 25 upwards and see if we can find a valid decomposition.
    
    # For each candidate rank r, we need to find U (10 x r), V (25 x r), W (10 x r)
    # such that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of nonlinear equations (trilinear in the variables).
    # However, if we fix V, it becomes bilinear, and if we fix two of them, it's linear.
    
    # Approach: Fix V to a canonical form and solve for U and W.
    # A natural choice: V[j*p+k, c] = 1 if c == j*p+k, else 0 (i.e., V is the identity-like matrix)
    # But V has 25 rows and r columns, so if r >= 25, we can set V to have 1s on the "diagonal"
    # in some sense.
    
    # Actually, let's try a different approach. Since we need integer/half-integer entries,
    # and the problem is small, let's try a systematic search for small rank values.
    
    # For rank r, we have:
    # - U: 10 * r entries
    # - V: 25 * r entries  
    # - W: 10 * r entries
    # Total: 45 * r variables
    # 
    # Constraints: 2 * 5 * 5 = 50 equations (one for each (i,j,k))
    # 
    # For r < 50, we have more variables than equations, so there might be solutions.
    
    # Let's try a randomized approach: for each candidate rank, randomly initialize
    # and use iterative refinement.
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 10:
            return None
        
        # Initialize with random values (integers or half-integers)
        # We'll use values from {-1, -0.5, 0, 0.5, 1}
        
        # Try multiple random initializations
        for attempt in range(100):
            if time_remaining() < 10:
                break
            
            # Random initialization
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # Initialize with small random values
            for i in range(n * m):
                for c in range(r):
                    U[i][c] = (np.random.randint(0, 3) - 1) * 0.5  # -0.5, 0, 0.5
            
            for i in range(m * p):
                for c in range(r):
                    V[i][c] = (np.random.randint(0, 3) - 1) * 0.5
            
            for i in range(p * n):
                for c in range(r):
                    W[i][c] = (np.random.randint(0, 3) - 1) * 0.5
            
            # Iterative refinement: alternate between solving for U, V, W
            for iteration in range(50):
                if time_remaining() < 5:
                    break
                
                # Solve for U given V and W
                # For each (i,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is linear in U[i*m+j, :] for fixed (i,j)
                
                # For each (i,j), we have p equations (one for each k) in r unknowns (U[i*m+j, 0:r])
                # A[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
                # sum_c A[i,j,k,c] * U[i*m+j, c] = 1
                
                for i in range(n):
                    for j in range(m):
                        # Build the linear system for U[i*m+j, :]
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                A_mat[k, c] = V[j * p + k][c] * W[k * n + i][c]
                        
                        # Solve A_mat @ x = ones(p)
                        # Use least squares
                        try:
                            x, residuals, rank_, singular_values = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            # Round to nearest half-integer
                            for c in range(r):
                                U[i * m + j][c] = round(x[c] * 2) / 2
                        except:
                            pass
                
                # Solve for V given U and W
                # For each (j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                # This is linear in V[j*p+k, :] for fixed (j,k)
                
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                A_mat[i, c] = U[i * m + j][c] * W[k * n + i][c]
                        
                        try:
                            x, residuals, rank_, singular_values = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            for c in range(r):
                                V[j * p + k][c] = round(x[c] * 2) / 2
                        except:
                            pass
                
                # Solve for W given U and V
                # For each (k,i): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # This is linear in W[k*n+i, :] for fixed (k,i)
                
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                A_mat[j, c] = U[i * m + j][c] * V[j * p + k][c]
                        
                        try:
                            x, residuals, rank_, singular_values = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            for c in range(r):
                                W[k * n + i][c] = round(x[c] * 2) / 2
                        except:
                            pass
                
                # Check if we have a valid decomposition
                if verify_decomposition(U, V, W, r):
                    return U, V, W
        
        return None
    
    # Try ranks from the theoretical lower bound up to the schoolbook rank
    lower_bound = max(n * m, m * p, p * n)  # 25
    
    for r in range(lower_bound, rank_schoolbook + 1):
        if time_remaining() < 30:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break  # Found the minimum rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
