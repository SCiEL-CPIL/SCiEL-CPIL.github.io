"""Matrix multiplication 4x5x5 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Use a structured sparse approach with greedy sparse recovery.
    We model the tensor as having a specific structure and try to find
    low-rank decompositions using random initialization + iterative refinement.
    
    The tensor T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    In our convention: W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] summed over r
    should equal delta for the correct (i,j,k) triple.
    
    Actually, the tensor decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    where T[i,j,k] = 1 for all valid (i,j,k) in the matrix multiplication.
    
    For 4x5x5: n=4, m=5, p=5
    U has shape (4*5, rank) = (20, rank)
    V has shape (5*5, rank) = (25, rank)  
    W has shape (5*4, rank) = (20, rank)
    
    The tensor has 4*5*5 = 100 non-zero entries (all 1s).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin before 150s kill
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p  # 100
    
    def schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions using structured approaches
    
    # Approach 1: Try various rank values and use random + refinement
    best_result = None
    best_rank = rank_schoolbook
    
    # Known lower bounds and upper bounds for 4x5x5
    # Strassen-like methods might help, but let's try computational approaches
    
    # Try structured decompositions with different patterns
    # Use a greedy approach: start with random rank-r matrices and try to fit
    
    def try_random_rank(r, num_trials=10, seed=42):
        """Try to find a rank-r decomposition using random initialization."""
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Random initialization with some structure
            U_np = rng.randn(n * m, r)
            V_np = rng.randn(m * p, r)
            W_np = rng.randn(p * n, r)
            
            # Normalize
            U_np = U_np / (np.linalg.norm(U_np, axis=0, keepdims=True) + 1e-10)
            V_np = V_np / (np.linalg.norm(V_np, axis=0, keepdims=True) + 1e-10)
            W_np = W_np / (np.linalg.norm(W_np, axis=0, keepdims=True) + 1e-10)
            
            # Check if this could work - compute residual
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for rr in range(r):
                            s += U_np[i * m + j, rr] * V_np[j * p + k, rr] * W_np[k * n + i, rr]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-6:
                # Convert to lists
                U_list = [[float(U_np[i, r]) for r in range(r)] for i in range(n * m)]
                V_list = [[float(V_np[i, r]) for r in range(r)] for i in range(m * p)]
                W_list = [[float(W_np[i, r]) for r in range(r)] for i in range(p * n)]
                return U_list, V_list, W_list
        
        return None
    
    # Try structured decompositions based on known algorithms
    # For 4x5x5, we can try to use block structure
    
    # Approach: Use the fact that 4x5x5 can be decomposed using
    # smaller matrix multiplication algorithms
    
    # Try rank values from low to high
    min_possible_rank = 20  # Lower bound heuristic
    max_try_rank = min(rank_schoolbook, 60)
    
    for r in range(min_possible_rank, max_try_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random trials for this rank
        result = try_random_rank(r, num_trials=3, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                best_rank = r
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    # If no improvement found, return schoolbook
    if best_result is None:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
