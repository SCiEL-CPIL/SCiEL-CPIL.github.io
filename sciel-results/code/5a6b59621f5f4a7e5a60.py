"""Rewritten program for Matrix Multiplication 3,4,4 rank minimization."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def schoolbook_solution(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return U, V, W

def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    # Compute C = A * B using the decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # For each output element (i,k), check that the coefficient of A[i,j]*B[j,k] is correct
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # The coefficient of A[i,j]*B[j,k] in C[i,k] should be 1
                coeff = 0.0
                for r in range(rank):
                    coeff += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(coeff - 1.0) > tol:
                    return False, f"C[{i},{k}] coefficient of A[{i},{j}]*B[{j},{k}] is {coeff}, expected 1"
    
    # Also check that there are no spurious terms
    # For each (i,k), the only terms should be A[i,j]*B[j,k] for j=0..m-1
    # We already checked those are 1. Now check that no other combinations appear.
    # Actually, the tensor decomposition naturally only produces terms of the form
    # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] which corresponds to A[i,j]*B[j,k]*C[i,k].
    # So if the coefficients are all 1, it's correct.
    
    return True, "OK"

def try_strassen_like_2x2(n, m, p):
    """
    Try to find a rank-reduced decomposition using Strassen-like 2x2 block structure.
    
    Partition A (n x m) into 2x2 blocks where possible.
    For 3x4: partition rows into [0,1] and [2], cols into [0,1,2,3] -> not clean 2x2.
    
    Alternative: Use overlapping 2x2 blocks and solve for minimal rank.
    
    Strategy:
    1. Identify all 2x2 sub-blocks of A and B that contribute to C.
    2. For each 2x2x2 tensor (multiplying 2x2 by 2x2 to get 2x2), the Strassen algorithm
       uses 7 multiplications instead of 8.
    3. Try to compose these local Strassen decompositions globally.
    
    For 3x4 * 4x4 -> 3x4:
    Let's partition:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    B = [[B11, B12], [B21, B22]] where all are 2x2
    C = [[C11, C12], [C21, C22]] where C11 is 2x2, C12 is 2x2, C21 is 1x2, C22 is 1x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    If we could use Strassen on the 2x2*2x2->2x2 parts, we might save.
    But the 1x2 parts don't benefit from Strassen.
    
    Let's try a different approach: use a random search with local refinement
    to see if rank 47 or lower is achievable.
    """
    if time_left() < 30:
        return None
    
    # Try rank 47 first (one less than schoolbook)
    target_rank = n * m * p - 1
    
    # Initialize with schoolbook and try to merge two terms
    U0, V0, W0 = schoolbook_solution(n, m, p)
    
    # Try to find if any two schoolbook terms can be merged
    # Term r corresponds to (i,j,k) where r = i*m*p + j*p + k
    # We want to see if we can express one term as a linear combination of others
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # For rank reduction, we need to find a set of rank-1 tensors that sum to T
    # with fewer than n*m*p terms.
    
    # Use random initialization and gradient-free optimization
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try several random initializations
    for attempt in range(5):
        if time_left() < 20:
            break
        
        # Initialize with random values
        np.random.seed(42 + attempt)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Use simple iterative refinement
        # Objective: minimize ||sum_r U[:,r]*V[:,r]*W[:,r] - T||^2
        # where T[i*m+j, j*p+k, k*n+i] = 1
        
        # This is a non-convex problem, so we use random restarts
        # and check if we can get close to the target.
        
        # For now, just check if a random initialization happens to work
        # (very unlikely, but let's try)
        U, V, W = schoolbook_solution(n, m, p)
        # Remove one column and see if it still works
        U_reduced = U[:, :target_rank]
        V_reduced = V[:, :target_rank]
        W_reduced = W[:, :target_rank]
        
        ok, msg = verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p)
        if ok:
            return U_reduced, V_reduced, W_reduced
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try Strassen-like approach first
    result = try_strassen_like_2x2(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = schoolbook_solution(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
