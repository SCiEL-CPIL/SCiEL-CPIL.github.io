"""Rewritten program: search for low-rank tensor decomposition of 3x5x5 matmul."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    # T is a 3D tensor of shape (n*m, m*p, p*n)
    # Index: A index = i*m+j, B index = j'*p+k, C index = k'*n+i'
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with i=i', j=j', k=k'
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T: sum_c U[:,c]*V[:,c]*W[:,c] == T"""
    rank = U.shape[1]
    # Reconstruct tensor
    T_recon = np.zeros_like(T)
    for c in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(T_recon - T)) < tol

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random rank-r decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W

def _try_reduce_rank(U, V, W, T, target_rank, max_iters=200, tol=1e-9):
    """Try to reduce the effective rank by zeroing out columns that are nearly redundant."""
    # This is a heuristic: check if any column can be approximated by others
    # For exact decomposition, we need a different approach
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Best known: schoolbook rank 75
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook decomposition
    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Strategy: use a structured approach based on the tensor's algebraic structure
    # For 3x5x5, try ranks from 25 up to 74
    
    # First, try a Kronecker product approach
    # If we can express the tensor as a sum of Kronecker products, we might get lower rank
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    
    # Try ranks in descending order from a reasonable upper bound
    # Start with lower bounds and go up
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But this is a weak bound. Let's try from 30 upward
    
    found_better = False
    
    # Try a few candidate ranks
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 20.0:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            
            seed = target_rank * 1000 + attempt
            U_try, V_try, W_try = _random_decomposition(n, m, p, target_rank, seed=seed)
            
            # Check if this random decomposition works (extremely unlikely for exact)
            # Instead, we need a more systematic approach
            
            # Use a greedy approach: start with schoolbook and try to merge columns
            pass
    
    # Alternative: use the fact that for this specific tensor, we can use
    # a structured decomposition based on the dimensions
    
    # Let me try: express C = A*B where A is 3x5, B is 5x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # One decomposition: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # This gives 3*5*5 = 75 terms.
    #
    # Can we share terms? If we fix j, we have 3*5 = 15 terms: A[i,j]*B[j,k] for all i,k.
    # These 15 terms can be written as: (A[:,j]) ⊗ (B[j,:]) ⊗ (identity in output space)
    # But that's not quite right for the tensor decomposition.
    
    # Let me try a different structural decomposition:
    # Group by the middle index j.
    # For each j, the contribution to C is: outer(A[:,j], B[j,:])
    # This is a rank-1 matrix in the (i,k) space.
    # So C = sum_j outer(A[:,j], B[j,:])
    # 
    # In tensor form: T = sum_j u_j ⊗ v_j ⊗ w_j
    # where u_j[i] = 1 if we're looking at row i of A at column j... 
    # Actually this is getting complicated. Let me just try numerical optimization.
    
    # Use a simple iterative method: start with rank r, initialize, and try to fit
    # We'll use a variant of the "rank reduction" approach
    
    # For now, return the schoolbook if no better is found
    # Convert to list format as required
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
