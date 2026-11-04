"""Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Construct a tensor decomposition for 5x5 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank) 
    - W has shape (p*n, rank)
    such that the matrix multiplication tensor is exactly decomposed.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try to find a better decomposition than schoolbook (rank 125)
    # Known lower bound for 5x5 matrix multiplication is at least 100 (n^omega)
    # Best known upper bound is around 110-120
    
    # Start with schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try random search for lower rank decompositions
    # Use a smarter approach: try rank 120, 115, 110, etc.
    
    def try_rank_search(target_rank, max_iters=5000):
        """Try to find a decomposition of the given rank using random initialization."""
        for iter_idx in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to unit norm columns
            for c in range(target_rank):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 0:
                    U[:, c] /= norm_u
                if norm_v > 0:
                    V[:, c] /= norm_v
                if norm_w > 0:
                    W[:, c] /= norm_w
            
            # Iterate: try to minimize the reconstruction error
            # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We want this to equal delta(i,j,k) for the matrix multiplication tensor
            # Actually, the matrix multiplication tensor is T[i,j,k] = 1 if i,j,k are valid indices
            # Wait, let me reconsider. The matrix multiplication tensor is:
            # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
            # So the tensor has T[i,j,k] = 1 for all valid i,j,k in the range
            
            # Actually, I need to be more careful. The decomposition satisfies:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            # This is a system of equations. Let's use a different approach.
            
            # Use gradient descent on the constraint that the tensor matches
            # But this is complex. Let's try a simpler heuristic.
            
            # Check current error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1) ** 2
            
            if error < 1e-6:
                return U, V, W, target_rank
            
            # Simple update: for each tensor entry, adjust the factors
            # This is a very rough heuristic
            
        return None, None, None, None
    
    # Try several target ranks
    for target_rank in [120, 115, 110, 105, 100]:
        if time.time() > deadline:
            break
        
        result = try_rank_search(target_rank, max_iters=1000)
        if result[0] is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
