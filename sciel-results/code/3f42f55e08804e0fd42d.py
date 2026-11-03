"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a decomposition with rank < 75
    # Using border rank relaxation: search for rank-74 with rational entries
    # where one factor is perturbed by an epsilon term
    
    def schoolbook_decomposition():
        rank = rank_schoolbook
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
    
    def tensor_error(U, V, W):
        """Compute the error in the tensor decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Should equal 1 if (i,j,k) is a valid index triple, else 0
        # Actually for matrix multiplication: T[i,j,k] = delta(i,j,k) where 
        # the product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 if we're computing the (i,k) entry using the (j) intermediate
        
        # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
        # has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 (this is the identity tensor in the middle dimension)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    # For matrix multiplication, T[i,j,k] should be 1
                    error += (val - 1.0) ** 2
        return error
    
    def to_fraction_list(mat):
        """Convert numpy array to list of Fractions."""
        return [[Fraction(int(x * 1000000), 1000000) for x in row] for row in mat]
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Start with schoolbook rank-75, try to reduce to rank-74
    # by finding a linear dependency among the rank-75 terms
    
    best_result = schoolbook_decomposition()
    best_rank = rank_schoolbook
    
    # Try border rank relaxation: perturb one factor and see if we can
    # express one term as a combination of others
    
    # For rank reduction from 75 to 74, we need to find coefficients c_r such that
    # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 where not all c_r are zero
    # and we can eliminate one term
    
    # Structured ansatz: group terms by (i,k) pairs
    # There are n*p = 15 output entries, each computed from m=5 intermediate sums
    # Total terms: 15 * 5 = 75
    
    # Try to find a dependency by considering the structure
    
    # Initialize with schoolbook
    U = [[0] * 75 for _ in range(n * m)]
    V = [[0] * 75 for _ in range(m * p)]
    W = [[0] * 75 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a rank-74 decomposition by searching for a linear dependency
    # Use random perturbations and check if the system becomes dependent
    
    random.seed(42)
    attempts = 0
    max_attempts = 5000
    
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        attempts += 1
        
        # Create a random perturbation of the schoolbook decomposition
        U_pert = [row[:] for row in U]
        V_pert = [row[:] for row in V]
        W_pert = [row[:] for row in W]
        
        # Perturb one column (term) slightly
        target_col = random.randint(0, 74)
        epsilon = random.uniform(0.01, 0.1)
        
        # Perturb the W factor for this column
        for idx in range(p * n):
            if W_pert[idx][target_col] != 0:
                W_pert[idx][target_col] += epsilon * random.uniform(-1, 1)
        
        # Now try to see if we can express column target_col as a combination
        # of the other 74 columns by solving a linear system
        
        # Build the tensor as a vector of length n*m*p = 75
        # Each column r defines a vector t_r in R^75 where t_r[i*m+j*p+k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        def tensor_vector(U_arr, V_arr, W_arr, r):
            vec = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[(i * m + j) * p + k] = U_arr[i * m + j][r] * V_arr[j * p + k][r] * W_arr[k * n + i][r]
            return vec
        
        # Get the tensor vectors for all columns
        tensors = np.array([tensor_vector(U_pert, V_pert, W_pert, r) for r in range(75)])
        
        # Check if column target_col is in the span of the others
        other_indices = [r for r in range(75) if r != target_col]
        A = tensors[other_indices].T  # 75 x 74
        b = tensors[target_col]
        
        # Solve least squares
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if residuals.size > 0 and residuals[0] < 1e-6:
                # Found a dependency! Now we can eliminate column target_col
                # New decomposition has rank 74
                new_U = [row[:target_col] + row[target_col+1:] for row in U]
                new_V = [row[:target_col] + row[target_col+1:] for row in V]
                new_W = [row[:target_col] + row[target_col+1:] for row in W]
                
                # Verify
                if verify_decomposition(new_U, new_V, new_W, tol=1e-8):
                    best_rank = 74
                    best_result = {"U": new_U, "V": new_V, "W": new_W}
                    return best_result
        except:
            pass
    
    # If no rank-74 found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
