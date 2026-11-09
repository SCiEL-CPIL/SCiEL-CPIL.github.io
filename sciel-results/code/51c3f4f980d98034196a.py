import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 105
    
    # Schoolbook decomposition: one column per (i, j, k) triple
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                # Deadline guard
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    # Attempt to find a lower rank decomposition using structured search
    # The matrix multiplication tensor for (n,m,p) = (3,5,7) has schoolbook rank 105.
    # We try to find a decomposition of rank < 105 using a structured approach.
    # 
    # Strategy: Use the fact that the tensor can be viewed as a linear map
    # and try to find a low-rank approximation using random linear combinations.
    # 
    # For small dimensions, we can try to find a decomposition of the form:
    # T = sum_{r=1}^{R} U_r * V_r * W_r
    # where U_r, V_r, W_r are rank-1 tensors.
    #
    # We use a greedy approach: start with the schoolbook decomposition and
    # try to merge terms using linear algebra.
    
    # Convert to numpy arrays for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find a lower rank decomposition using random linear combinations
    # We attempt to find a decomposition of rank R < 105
    
    best_rank = rank_schoolbook
    best_U = U
    best_V = V
    best_W = W
    
    # Try various target ranks
    for target_rank in range(100, 104):  # Try ranks 100-104
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of the given target rank
        # We use a randomized approach with structured initialization
        
        # Initialize with random linear combinations of the schoolbook terms
        np.random.seed(42)
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Create random weight matrices
            weights = np.random.randn(target_rank, rank_schoolbook)
            
            # Compute new factor matrices
            U_trial = U_np @ weights.T  # shape (n*m, target_rank)
            V_trial = V_np @ weights.T  # shape (m*p, target_rank)
            W_trial = W_np @ weights.T  # shape (p*n, target_rank)
            
            # Verify the decomposition
            # The tensor should satisfy: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # For the schoolbook decomposition, T[i,j,k] = 1 if i,j,k are valid indices
            
            # Check a few entries to verify correctness
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0
                        actual = 0.0
                        for r in range(target_rank):
                            actual += U_trial[i*m+j, r] * V_trial[j*p+k, r] * W_trial[k*n+i, r]
                        if abs(actual - expected) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_trial.tolist()
                best_V = V_trial.tolist()
                best_W = W_trial.tolist()
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better decomposition, use it
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise, return the schoolbook decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
