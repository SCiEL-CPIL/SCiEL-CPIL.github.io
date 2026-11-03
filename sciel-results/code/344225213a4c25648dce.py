"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The matrix multiplication tensor for (n x m) * (m x p) has rank at most n*m*p
    # For small matrices, we can try to find lower rank via random search or structured search
    
    # The tensor T[i,j,k] corresponds to the (i,k) entry of A*B where A is n x m and B is m x p
    # T has shape (n, p, m) with T[i,k,j] = 1 if the (i,k) output depends on A[i,j]*B[j,k]
    # Actually, the standard tensor for matmul: T[i,k,j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
    # Wait, let me be more careful.
    
    # The matrix multiplication tensor T is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[i,k,j] = 1 for all valid i,k,j
    # This is a 3-tensor of shape (n, p, m) = (3, 6, 3)
    
    # A rank-r decomposition means:
    # T[i,k,j] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  ... wait, need to match the interface
    
    # Interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The decomposition should satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to saying the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # can be written as sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Let's try random search for lower ranks
    # Start from rank_schoolbook - 1 and go down
    
    # Actually, let's think about known lower bounds.
    # For 3x3 * 3x3, the rank is known to be at most 23 (Strassen-like) and at least 19.
    # For 3x3 * 3x6, it's less well-known but likely similar per output element.
    
    # Let's try a random search approach with verification
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition is exact."""
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try random search for lower ranks
    # Use a strategy: start with schoolbook and try to reduce
    
    # Actually, for this specific problem, let's try a more structured approach.
    # The key insight is that we can use bilinear algorithms.
    
    # Let's try random initialization and gradient-free optimization
    # But first, let's see if we can find any improvement at all.
    
    # Try ranks from schoolbook-1 downward
    for rank in range(rank_schoolbook - 1, max(15, rank_schoolbook - 10), -1):
        if time.time() > deadline - 10:
            break
            
        # Try several random initializations
        max_attempts = 5 if rank > 30 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            random.seed(attempt * 1000 + rank)
            np.random.seed(attempt * 1000 + rank)
            
            # Use small random values
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Verify
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank, n, m, p):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        # If we found a valid decomposition at this rank, try lower
        if best_rank == rank:
            continue
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
# EVOLVE-BLOCK-END
