import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s to stay safely under 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # The matrix multiplication tensor T has indices (i, j, k) where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i, j, k] = 1 for all i, j, k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # Structural approach: exploit the fact that n=2 is small.
    # For each pair (j, k), we need the vector over i to be constant (1, 1).
    # This means for each (j,k), sum_r U[i, r] * V[j,k, r] * W[k,i, r] = 1 for i=0,1.
    
    # Let's try a structured search for rank < 36.
    # Key insight: if we can find a decomposition where some terms cancel or combine,
    # we might reduce the rank.
    
    # Try random search with integer/half-integer entries for lower ranks.
    # Given the time limit, we'll try ranks from 35 down to maybe 20.
    
    import random
    random.seed(42)
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def random_decomposition(rank, time_budget=1.0):
        """Try to find a random decomposition of given rank."""
        t_start = time.time()
        for attempt in range(100):
            if time.time() - t_start > time_budget:
                break
            
            # Generate random entries from {-1, -0.5, 0, 0.5, 1}
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W, rank):
                return U, V, W
        return None, None, None
    
    # Try structured approach: fix some entries to exploit symmetry
    # For n=2, we have two rows. Let's see if we can use a Strassen-like trick.
    
    # Actually, let's try a more systematic approach.
    # The tensor is all-ones. We need rank-1 tensors that sum to all-ones.
    
    # Try small ranks with more careful search
    for target_rank in range(35, 20, -1):
        if time.time() > deadline - 10:
            break
        
        result = random_decomposition(target_rank, time_budget=0.5)
        if result[0] is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    # If we haven't found anything better, try a structured approach
    # based on the observation that for n=2, the i-index only has 2 values.
    
    # Let's try to construct a decomposition by grouping terms.
    # For each (j, k), we need two equations (for i=0 and i=1).
    # This gives us 2*m*p = 36 equations total, but they're not independent.
    
    # Try a specific structured pattern
    if time.time() < deadline - 5:
        # Try rank 35: remove one schoolbook term and see if we can compensate
        # This is unlikely to work with simple adjustments, but let's try.
        
        # Alternative: try to find a decomposition using the fact that
        # the tensor has a lot of structure.
        
        # For now, if we haven't improved, return the schoolbook result.
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
