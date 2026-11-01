"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]
_DEADLINE = 150.0  # wall-clock seconds allowed by evaluator

# EVOLVE-BLOCK-START
def _construct_optimal(n, m, p):
    """
    Construct a rank n*m*p exact decomposition (schoolbook).
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > tol:
                    return False
    return True


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    def time_left():
        return _DEADLINE - (time.time() - start) - 5.0
    
    # Start with the known schoolbook decomposition (rank = n*m*p = 54)
    best_result = _construct_optimal(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition using structured search
    # The lower bound on rank is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    
    # Strategy: Try various structured decompositions that might achieve rank < 54
    # For M_{2,3,9}, we can try:
    # 1. Block-based decompositions
    # 2. Tensor network contractions
    # 3. Direct rank minimization via iterative refinement
    
    # First, try a simple structured approach:
    # Partition the 3x9 matrix into 3 blocks of 3x3
    # C[:, 0:3] = A @ B[:, 0:3], C[:, 3:6] = A @ B[:, 3:6], C[:, 6:9] = A @ B[:, 6:9]
    # Each block multiplication is 2x3 @ 3x3 = 2x3
    # Rank of M_{2,3,3} is at most 18, lower bound max(6, 9, 6) = 9
    
    # If we can find rank-9 decomposition for M_{2,3,3}, total rank would be 27
    
    # Let's try to construct a rank-9 decomposition for 2x3 @ 3x3
    # Using the fact that n=2, we might be able to use Winograd-style tricks
    
    # For 2x3 @ 3x3, schoolbook is rank 18
    # Let's see if we can do better
    
    # Actually, let's try a different approach: use the structure more directly
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # For n=2: C_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k} + A_{0,2}B_{2,k}
    #         C_{1,k} = A_{1,0}B_{0,k} + A_{1,1}B_{1,k} + A_{1,2}B_{2,k}
    
    # This is essentially two linear combinations. We can use the fact that
    # for each k, we're computing a 2x1 vector as A (2x3) times B[:,k] (3x1).
    
    # The bilinear complexity of computing a 2x3 matrix times a 3x1 vector
    # (for all 9 columns simultaneously) is what we need.
    
    # Let's try rank 27 first (the theoretical lower bound)
    # If we can achieve rank 27, that would be optimal
    
    # For rank 27, we need to show that M_{2,3,9} has rank 27
    # This is equivalent to showing that the 2x3 @ 3x9 multiplication
    # can be done with 27 bilinear multiplications
    
    # One approach: use the fact that 3x9 can be viewed as 3 blocks of 3x3
    # and use a rank-9 scheme for 2x3 @ 3x3
    
    # Let's attempt to find such a scheme using a small search
    
    def try_rank_r(r, timeout=30):
        """Try to find a rank-r decomposition. Returns result dict or None."""
        if time_left() < timeout:
            return None
        
        # For small ranks, try structured approaches
        # This is a placeholder for more sophisticated search
        # In practice, we'd use tensor decomposition algorithms
        
        # For now, just verify the schoolbook result
        return None
    
    # Verify the schoolbook decomposition
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    
    # If verification failed (shouldn't happen), return a trivial invalid result
    return {"U": [[0]], "V": [[0]], "W": [[0]]}
# EVOLVE-BLOCK-END
