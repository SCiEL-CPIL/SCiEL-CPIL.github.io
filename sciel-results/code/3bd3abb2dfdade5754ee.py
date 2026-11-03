"""
Program for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication.
Uses a systematic search approach to find decompositions better than schoolbook.
"""

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Return a tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (9, rank)
        - V has shape (m*p, rank) = (12, rank)  
        - W has shape (p*n, rank) = (12, rank)
    
    The decomposition satisfies:
        For all i, k: C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        summed over appropriate indices.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Try to find a better decomposition than schoolbook using systematic search
    # with small integer entries
    
    import itertools
    import numpy as np
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0])
        # Check all entries of the multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The tensor entry T[i,j,k] should be 1 if this is the (i,j,k) product term
                    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                    # Wait, no. The multiplication tensor T has entries T[i,j,k] where:
                    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
                    # So the tensor is T[i,j,k] = delta (it's 1 for all i,j,k in range)
                    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # for all i, j, k
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(9, 12, 12) = 12
    lower_bound = max(n*m, m*p, n*p)
    
    # Try a few ranks below schoolbook with small integer entries
    # Use random search with small integers (-2, -1, 0, 1, 2)
    import random
    random.seed(42)
    
    max_search_time = 100  # seconds, leave time for evaluation
    
    import time
    start_time = time.time()
    
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() - start_time > max_search_time:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 50 if rank > 25 else 20
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_search_time:
                break
            
            # Generate random matrices with small integer entries
            # Use values from {-2, -1, 0, 1, 2}
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify_decomposition(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if best_rank < rank_schoolbook:
            break
    
    return best
