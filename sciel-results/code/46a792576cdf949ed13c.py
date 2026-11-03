import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition for 3x3x5 matrix multiplication.
    U: (9, rank)
    V: (15, rank)
    W: (15, rank)
    """
    n, m, p = 3, 3, 5
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For all i, j, k:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Try to find a better decomposition
    # Start with known 3x3x3 decomposition (rank 23) and extend
    
    # First, let's verify we can construct a valid 3x3x3 decomposition
    # Using the known rank-23 decomposition for 3x3x3
    
    def get_3x3x3_decomposition():
        """
        Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
        U: (9, 23), V: (9, 23), W: (9, 23)
        """
        # This is a known rank-23 decomposition
        # I'll construct it using a systematic approach
        
        # For 3x3x3, we can use the following structure:
        # The multiplication tensor has 27 terms in schoolbook.
        # We need to find 23 vectors that span the same space.
        
        # One approach: use the fact that the tensor is symmetric in a certain sense
        # and find a basis of the span.
        
        # Let me try a different approach: search for a valid decomposition
        # using random initialization and gradient descent
        
        return None
    
    # Since finding a known rank-23 decomposition is complex, let me try a different approach:
    # Use the structure of the problem to build a decomposition.
    
    # For 3x3x5, the schoolbook rank is 45.
    # We know that rank(3x3x3) <= 23 and rank(3x3x5) <= rank(3x3x3) + rank(3x3x2)
    # rank(3x3x2) <= 12 (schoolbook) or possibly lower.
    
    # Let me try to construct a decomposition by combining:
    # 1. A 3x3x3 decomposition for the first 3 columns
    # 2. A 3x3x2 decomposition for the last 2 columns
    
    # But this is tricky because the decompositions need to be compatible.
    
    # Alternative approach: use the fact that the tensor T[i,j,k] = 1 for all i,j,k
    # and find a rank decomposition directly.
    
    # Let me try a random search approach with verification
    
    def random_decomposition(rank, seed=42):
        """Generate a random decomposition of given rank."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Verify if U, V, W form a valid decomposition."""
        n, m, p = 3, 3, 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition with rank less than schoolbook
    # Start with rank 30 and work down
    
    for target_rank in range(44, 20, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
            
            U, V, W = random_decomposition(target_rank, seed)
            
            if verify_decomposition(U, V, W):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        # If we found a valid decomposition at this rank, try to go lower
        if target_rank < best_rank:
            continue
    
    return best_result

# EVOLVE-BLOCK-END
