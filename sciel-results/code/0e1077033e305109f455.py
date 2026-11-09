"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 8 matrix."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matrix multiplication tensor T[i,j,k] = delta for multiplying (i,j) by (j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Actually, the tensor for matrix multiplication is:
    # T[i, j, k] = 1 if we're multiplying A[i,j] * B[j,k] to get C[i,k]
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    rank = U.shape[1]
    # Build the tensor T from the factorization
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Use einsum for efficiency
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The target tensor is all ones for valid (i,j,k) combinations
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    target = np.ones((n, m, p))
    
    return np.allclose(T, target, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
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

def _try_random_search(n, m, p, max_rank, time_limit, rng):
    """Try to find a decomposition with rank <= max_rank using random search."""
    deadline = time.time() + time_limit
    
    for rank in range(1, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random attempts for each rank
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers or zeros
            U = rng.randint(-2, 2, size=(n * m, rank)).tolist()
            V = rng.randint(-2, 2, size=(m * p, rank)).tolist()
            W = rng.randint(-2, 2, size=(p * n, rank)).tolist()
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Also try with just 0s and 1s
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            U = rng.randint(0, 1, size=(n * m, rank)).tolist()
            V = rng.randint(0, 1, size=(m * p, rank)).tolist()
            W = rng.randint(0, 1, size=(p * n, rank)).tolist()
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_search(n, m, p, max_rank, time_limit, rng):
    """Try structured approaches to find lower rank decompositions."""
    deadline = time.time() + time_limit
    
    # For matrix multiplication, known bounds:
    # 2x3 * 3x8: The rank is at most 48 (schoolbook)
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also, the rank must be at least the max of the three dimensions' products
    
    # Try rank values from lower bound up
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, min(max_rank, lower_bound + 20) + 1):
        if time.time() > deadline:
            break
        
        # Try random search with this specific rank
        result = _try_random_search(n, m, p, rank, time.time() - time.time() + 5, rng)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    deadline = time.time() + 120  # 120 second limit with margin
    
    # Use a fixed seed for reproducibility
    rng = random.Random(42)
    
    # Try to find a better (lower rank) decomposition
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Known results suggest the rank might be close to this lower bound
    
    # First, try random search for ranks between lower bound and schoolbook
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a range of ranks, starting from the lower bound
    search_time_per_rank = 2.0  # seconds per rank value
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:  # Keep 10s margin
            break
        
        time_budget = min(search_time_per_rank, deadline - time.time() - 5)
        if time_budget <= 0:
            break
        
        result = _try_random_search(n, m, p, rank, time_budget, rng)
        if result is not None:
            best_result = result
            best_rank = rank
            # If we found a solution at the lower bound, we can stop
            if rank == lower_bound:
                break
    
    return best_result
