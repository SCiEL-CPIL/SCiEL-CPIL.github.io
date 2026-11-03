import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    Uses a slice-and-combine strategy: decompose into 5 slices of 3x3x1,
    then find a unified rank-r decomposition with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank_school = n * m * p  # 45
    
    deadline = time.time() + 140  # 10s safety margin
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition with integer/half-integer entries."""
        # Check integer/half-integer constraint
        for matrix in [U, V, W]:
            for val in matrix.flatten():
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
        # Check the tensor condition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def make_schoolbook():
        """Create the schoolbook decomposition."""
        rank = rank_school
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_rank_r(r, max_iters=20000, tol=1e-9):
        """
        Try to find a rank-r decomposition using iterative refinement.
        Uses a structured initialization based on slice decomposition.
        """
        if r >= rank_school:
            return make_schoolbook()
        
        # Initialize with a structured pattern
        # For each of the 5 slices, use a 3x3x3-style decomposition
        # Then try to merge them
        
        # Strategy: Start with schoolbook and try to reduce rank by
        # finding linear combinations that preserve the tensor
        
        # Initialize U, V, W with random half-integer values
        np.random.seed(42)
        
        # Create a structured initialization
        # For rank r, we can try to use r basis tensors
        # Each basis tensor is a rank-1 tensor U_r ⊗ V_r ⊗ W_r
        
        # Let's try a parameterized approach:
        # Assume U has the form where each row is a linear combination of basis vectors
        # Similarly for V and W
        
        # For now, try a simple iterative approach:
        # 1. Start with a feasible solution (schoolbook)
        # 2. Try to reduce rank by merging columns
        
        # Alternative: Use the fact that the tensor is separable across k
        # T[i,j,k] = 1 for all i,j,k
        # This means the tensor is the outer product of three all-ones vectors
        # T = ones(n,m) ⊗ ones(m,p) ⊗ ones(p,n) -- no, that's not right
        
        # Actually T[i,j,k] = 1 for all i,j,k is just the all-ones tensor
        # So we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        # This is a system of 45 equations in r*9 + r*15 + r*15 = 39r unknowns
        # For r=7: 273 unknowns, 45 equations -- underdetermined
        
        # Let's try to solve this directly using a structured ansatz
        
        # Ansatz: Use the same U for all slices, and let V and W depend on k
        # U[i,j,r] is shared across all k
        # V[j,k,r] and W[k,i,r] vary with k
        
        # For a given r, we need:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
        
        # Let's try r=7,8,9,... and see if we can find a solution
        
        # Use a simple iterative method:
        # Initialize U, V, W with small random values
        # Then iteratively update to satisfy the constraints
        
        # Better: Use the fact that we can set U to be a fixed matrix
        # and solve for V and W
        
        # Let's try: fix U to be a 9xr matrix with entries in {-1, 0, 1}
        # Then for each k, solve for V[:,k,:] and W[k,:] such that
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        # This is still nonlinear. Let's try a different approach.
        
        # Approach: Use the schoolbook and try to find linear dependencies
        # between the 45 basis tensors to reduce the rank
        
        # The 45 basis tensors are e_i ⊗ e_j ⊗ e_k for i in [0,3), j in [0,3), k in [0,5)
        # We want to find the minimum number of rank-1 tensors whose sum equals the all-ones tensor
        
        # The all-ones tensor J = ones(3,3,5)
        # We want to write J as a sum of r rank-1 tensors
        
        # A rank-1 tensor is u ⊗ v ⊗ w where u in R^9, v in R^15, w in R^15
        # (using the flattened indices)
        
        # Actually, let's use a more direct approach:
        # Try to find U (9xr), V (15xr), W (15xr) with half-integer entries
        # such that the tensor condition is satisfied
        
        # Use a randomized approach with local search
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random half-integer values
            scale = 1.0
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(9, r))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(15, r))
            W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(15, r))
            
            # Compute the error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_init[i*m+j, :] * V_init[j*p+k, :] * W_init[k*n+i, :])
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
            
            if err < tol:
                break
        
        if best_U is not None and best_err < tol:
            # Verify and return
            if check_decomposition(best_U, best_V, best_W):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return None
    
    # Try to find a lower-rank decomposition
    # Start from rank 7 (Strassen for 3x3x3) and go up
    for r in range(7, min(20, rank_school)):
        if time.time() > deadline - 30:
            break
        
        result = try_rank_r(r)
        if result is not None:
            return result
    
    # If we couldn't find a lower-rank decomposition, return schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
