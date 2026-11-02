"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (i,j,k) combinations where indices don't align: sum should be 0
    # Actually the tensor T[i,j,k] = delta_{i,j,k} for matrix mult: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) in [0,n)x[0,m)x[0,p)
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, deadline):
    """Try to find a decomposition of rank target_rank by structured search."""
    # Start from schoolbook and try to merge/remove columns
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p
    
    # Strategy: try random subsets of columns of size target_rank
    # and check if they form a valid decomposition
    # This is very hard, so use a smarter approach:
    
    # Try structured patterns: for each (i,j,k) triple, assign it to a column
    # but allow multiple triples to share a column if their "signature" matches
    
    # A better approach: use the fact that for matrix multiplication,
    # we can sometimes share columns between different (i,j,k) triples
    # if the row/column indices align properly.
    
    # Let's try: group (i,j,k) by some invariant and see if we can merge
    # For example, if two triples (i1,j1,k1) and (i2,j2,k2) have i1=i2 and k1=k2,
    # they might be mergeable (but j1 != j2 typically)
    
    # Actually, the standard approach for small matrices is to use known results.
    # For 4x5x5, the known rank is around 100 (schoolbook) but might be lower.
    
    # Let's try a systematic search: start with schoolbook and try to eliminate
    # columns by checking if they are linearly dependent in some sense.
    
    # Simpler: try random perturbations and see if we can reduce rank
    # This is unlikely to work well, so let's just return schoolbook if we can't do better.
    
    # Try a few structured patterns
    patterns = []
    
    # Pattern 1: Try to use fewer columns by sharing across j indices
    # For fixed i,k, the contribution is sum_j A[i,j]*B[j,k]
    # If we can express this as a product of functions of i and k, we save rank
    
    # This is getting complex. Let's just do a time-limited random search.
    best_result = None
    best_rank = rank
    
    # Try to find a lower rank by random column selection
    num_attempts = 1000
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
        
        # Randomly select target_rank columns from schoolbook
        cols = random.sample(range(rank), target_rank)
        U_sub = [row[c] for row in U for c in [0]*len(cols)]  # This is wrong, let me fix
        
        # Actually, let me construct properly
        U_new = []
        for row in U:
            U_new.append([row[c] for c in cols])
        V_new = []
        for row in V:
            V_new.append([row[c] for c in cols])
        W_new = []
        for row in W:
            W_new.append([row[c] for c in cols])
        
        if _verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, target_rank
    
    return None, None, None, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank
    # Known lower bounds: for n x m x p matrix mult, rank >= max(n*m, m*p, n*p) / something
    # Actually, rank >= n*m*p / max(n,m,p) is not right.
    # The rank is at least the dimension of the tensor space, which is n*m*p.
    # Wait, no. The rank can be less than n*m*p.
    
    # For 4x5x5, let's try ranks from 99 down to some lower bound
    # Lower bound: the tensor has n*m*p = 100 non-zero entries, but they might be dependent.
    # A simple lower bound: rank >= max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # But that's very weak.
    
    # Let's try a few target ranks
    for target_rank in range(best_rank - 1, max(best_rank - 30, 20), -1):
        if time.time() > deadline - 5:
            break
        result = _try_rank_reduction(n, m, p, target_rank, deadline - 5)
        if result[0] is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
