"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy: Try to find a lower rank by exploiting the structure
    # For each (i,k), we have m terms (one for each j). We can try to merge
    # pairs of j-terms using linear combinations.
    
    # Approach: For a fixed (i,k), try to represent the m terms using fewer
    # rank-1 terms by finding linear dependencies.
    
    # Let's try a systematic approach:
    # 1. Start with schoolbook
    # 2. For each pair of j values (j1, j2) and fixed (i,k), see if we can
    #    merge them into a single term
    # 3. This requires solving a system of equations
    
    # Actually, let's try a different approach: use the fact that the tensor
    # has a specific structure and try to find a basis.
    
    # For now, let's try to improve the schoolbook by merging terms that
    # have the same (i,k) but different j.
    
    # Key insight: For fixed i and k, we need sum_j U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    # We can try to find a set of rank-1 terms that satisfy this for all (i,k).
    
    # Let's try a greedy approach: start with schoolbook, then try to merge
    # pairs of terms.
    
    def try_merge(U, V, W, r1, r2, i, k):
        """Try to merge terms r1 and r2 for fixed (i,k). Returns True if successful."""
        # This is complex; for now, just return False
        return False
    
    # Let's try a simpler approach: use random search with a better initialization
    # that respects the structure.
    
    np.random.seed(42)
    
    # Try to find a rank-104 decomposition
    # Strategy: Start with schoolbook, then try to perturb it slightly
    
    for target_rank in [104, 103, 102, 101, 100]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook-like structure
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # Fill with schoolbook terms for the first target_rank columns
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U_r[i*m+j][col] = 1
                        V_r[j*p+k][col] = 1
                        W_r[k*n+i][col] = 1
                        col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            # Now try to adjust the last few terms to satisfy the remaining constraints
            # This is a linear system problem
            
            # For each (i,j,k) that wasn't covered, we need to adjust
            # Let's try to solve this using least squares
            
            # Actually, this is getting complex. Let's try a different approach.
            # Use a local search: start with schoolbook, then try to remove one term
            # and adjust the others.
            
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
