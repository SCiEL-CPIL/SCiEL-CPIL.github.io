"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i in range(n) = range(3)  (row of A, row of C)
    - j in range(m) = range(3)  (col of A, row of B)
    - k in range(p) = range(8)  (col of B, col of C)
    
    T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    A tensor decomposition of rank r means:
    T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, let me re-read the interface:
    U has shape (n*m, rank)
    V has shape (m*p, rank)  
    W has shape (p*n, rank)
    
    And the output factor follows the official trace convention W[k*n+i, r].
    
    So the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This is the standard matrix multiplication tensor decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Schoolbook gives rank n*m*p = 3*3*8 = 72
    # We need to find if we can do better.
    
    # For matrix multiplication, the rank is known to be at least n*p*max(n,m)/something...
    # Actually, for M_{n,m,p}, the rank is at least max(n*m, m*p, p*n) and at most n*m*p.
    
    # For 3x3 * 3x8:
    # Lower bounds: max(9, 24, 24) = 24
    # Upper bound: 72 (schoolbook)
    
    # Let's try to find a decomposition with rank < 72 using random search.
    # 
    # Strategy: Try random rank-r decompositions and check if they exactly decompose T.
    # 
    # Actually, a better approach: use the fact that for small matrices, we can use
    # known results. The rank of M_{3,3,8} is known to be... let me think.
    #
    # Actually, let's just try random search with various ranks starting from lower bound.
    
    def is_valid_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For the matrix multiplication tensor, T[i,j,k] = 1 for all (i,j,k) in the "slice"
        # Actually, T[i,j,k] = 1 if the triple (i,j,k) corresponds to a term in the product.
        # For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is T[i,j,k] = 1 for all valid (i,j,k), meaning each term A[i,j]*B[j,k] contributes to C[i,k].
        
        # The tensor T is defined as T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because each such triple represents one multiplication-addition step.
        
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # And also, we need to ensure that the decomposition is "exact" for matrix multiplication.
        # Actually, the condition is that the tensor equals the matrix multiplication tensor.
        
        # Let's verify by checking a few entries.
        # Actually, let's check all entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions using random search
    # For each rank r from lower_bound to best_rank-1, try random initializations
    
    # Lower bound for rank of M_{n,m,p} is at least max(n*m, m*p, p*n)
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward, but limit the search
    # Use random Gaussian initializations and check if the decomposition is exact.
    
    # Actually, random search for exact decompositions is very hard. Let me think of a better approach.
    
    # One approach: use the fact that we can parameterize the decomposition and use optimization.
    # But exact solutions are hard to find via optimization.
    
    # Alternative: use known algebraic techniques. For M_{3,3,8}, let's see if we can find
    # a structured decomposition.
    
    # Let's try a simple approach: assume U, V, W have entries from {0, 1, -1} or small rationals.
    # Try random sparse decompositions.
    
    import random
    
    def try_random_decomposition(rank, n_trials=1000):
        """Try to find a rank-r decomposition with entries in {-1, 0, 1} or small integers."""
        n, m, p = DIMENSIONS
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with small integers
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                # Each term (u, v, w) should correspond to a "pattern"
                # For matrix multiplication, each term typically has a specific structure.
                # Let's try random assignments.
                for idx in range(n * m):
                    if random.random() < 0.3:
                        U[idx, c] = random.choice([-1, 1])
                for idx in range(m * p):
                    if random.random() < 0.3:
                        V[idx, c] = random.choice([-1, 1])
                for idx in range(p * n):
                    if random.random() < 0.3:
                        W[idx, c] = random.choice([-1, 1])
            
            if is_valid_decomposition(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try a few ranks
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(rank, n_trials=500)
        if result is not None:
            best_rank = rank
            best_U, best_V, best_W = result
            break  # Found a valid decomposition, try to find even lower
    
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
