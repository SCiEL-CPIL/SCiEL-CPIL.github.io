import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Try ranks from a reasonable upper bound down to the theoretical minimum
    # For 2x5x5, schoolbook is 50, but we can do much better
    # Start with rank 25 (half of schoolbook) and work down
    
    best_result = None
    best_rank = None
    
    # The tensor T[i,j,k] = 1 if i,j,k are compatible
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,k,j] = delta_{compatible}
    
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try to find a decomposition with rank r
    # Use a greedy/iterative approach: for each rank, try to find U, V, W
    
    for rank in range(25, 5, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with random sparse matrices
            np.random.seed(attempt * 1000 + rank)
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Set random non-zero entries
            for c in range(rank):
                # Each column of U, V, W should have some structure
                # For matrix multiplication, we can use the fact that
                # each "multiplication" corresponds to a rank-1 term
                
                # Randomly select which entries are non-zero
                u_idx = np.random.choice(n * m, size=3, replace=False)
                v_idx = np.random.choice(m * p, size=3, replace=False)
                w_idx = np.random.choice(p * n, size=3, replace=False)
                
                U[u_idx, c] = 1
                V[v_idx, c] = 1
                W[w_idx, c] = 1
            
            # Check if this decomposition is valid
            if _check_decomposition(U, V, W, n, m, p):
                found = True
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            # Found a valid decomposition, try lower ranks
            continue
        else:
            # If we can't find a valid one at this rank, try lower
            # But we need to be careful - maybe rank is too low
            pass
    
    # If no better found, return schoolbook
    if best_result is None:
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
    
    return best_result

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,k,j] should equal delta for compatible indices
    # For matrix multiplication: the tensor is T[i,k,j] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # Actually, the standard tensor for matrix multiplication M_{n,m,p} has:
    # T[i, k, j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
    # This represents C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The decomposition should satisfy:
    # T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True
