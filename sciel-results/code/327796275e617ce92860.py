"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    Searches for a rank-40 decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave margin for safety
    
    # Schoolbook rank 48
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook is valid
    def verify_decomposition(U, V, W, n, m, p):
        rank = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if True else 0  # All should be 1
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - 1) > 1e-10:
                        return False
        # Check that no spurious terms appear
        # For this problem, we need: sum_r U[i*m+j,r]*V[j'*p+k',r]*W[k''*n+i'',r] = 0
        # unless (i,j,k) = (i',j',k')
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i, j, k) == (i2, j2, k2):
                                    continue
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i * m + j, r] * V[j2 * p + k2, r] * W[k2 * n + i2, r]
                                if abs(actual) > 1e-10:
                                    return False
        return True
    
    # Try to find rank-40 by using a structured approach
    # Key idea: For n=2, we can potentially share structure between the two output rows
    
    # Let's try a specific construction:
    # Use 40 terms where we exploit the small n dimension
    
    # Approach: Start with rank 48, try to remove 8 columns by finding
    # linear dependencies or using a different basis
    
    # For now, return schoolbook if search fails
    best_solution = (U_school, V_school, W_school, schoolbook_rank)
    
    # Try random search for rank 40
    for target_rank in [40, 41, 42, 43, 44, 45, 46, 47]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values from {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, (n * m, target_rank))
            V_rand = np.random.randint(-1, 2, (m * p, target_rank))
            W_rand = np.random.randint(-1, 2, (p * n, target_rank))
            
            if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_solution = (U_rand, V_rand, W_rand, target_rank)
                found = True
                break
        
        if found:
            break
    
    # Also try a structured approach for rank 40
    if best_solution[3] > 40 and time.time() < deadline - 15:
        # Try using the structure: for each j in 0..2, k in 0..7
        # We have 2 outputs (i=0,1)
        # Total 24 (j,k) pairs, each needs 2 outputs = 48 scalar products
        # We can use 40 terms by sharing
        
        # Construction idea:
        # For j=0: use terms 0-13 (14 terms)
        # For j=1: use terms 14-27 (14 terms)  
        # For j=2: use terms 28-41 (14 terms)
        # But we only have 40, so we need to share 4 terms
        
        # Let me try a different structured approach
        # Use 5 terms per (j,k) pair for some, 4 for others
        
        # Actually, let's just try to verify if a rank-40 solution exists
        # by using a more systematic search with constraints
        
        pass
    
    U, V, W, rank = best_solution
    
    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
