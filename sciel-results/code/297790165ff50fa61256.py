"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 2*2*5 = 20
    
    # Lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb, r_sb)
    
    # Try to find better decompositions
    # Strategy: Use structured search with small integer/half-integer entries
    
    # For 2x2 * 2x5, we can think of it as:
    # A (2x2) * B (2x5) = C (2x5)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    
    # Let's try ranks from 11 to 19 (lower bound is 10)
    # Use a smarter search: fix some structure and vary the rest
    
    # Approach 1: Try random constructions with small integer entries
    values = [-2, -1, 0, 1, 2]
    
    for target_rank in range(11, schoolbook_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 10000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V, W with entries from values
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p, target_rank):
                found = True
                best_decomp = (U, V, W, target_rank)
                break
        
        if found:
            break
    
    # Approach 2: Try structured constructions
    # For 2x2 * 2x2, Strassen gives rank 7
    # For the extra 3 columns, we might be able to share some computations
    
    # Let's try a hybrid approach: use Strassen-like for the first 2 columns,
    # and add terms for the remaining 3 columns
    
    # Actually, let's try a more systematic search with constrained structure
    # Fix U and V to have a specific pattern, and solve for W
    
    # For now, if we haven't found anything better, return the best we have
    U, V, W, rank = best_decomp
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
