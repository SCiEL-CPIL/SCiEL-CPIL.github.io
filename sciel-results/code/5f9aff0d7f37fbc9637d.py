import time
import numpy as np
from itertools import product

# Public task contract
# Matrix multiplication 3,3,6
# Output factor W[k*n+i, r] (trace convention)

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """
    Verify that U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    n=3, m=3, p=6
    """
    n, m, p = 3, 3, 6
    rank = len(U[0])
    
    # Check dimensions
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    # Compute the tensor T[a, b, d] = sum_c U[a,c] V[b,c] W[d,c]
    # We only need to check that T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and that all other entries are 0 (but we can just check the required ones are 1
    # and a random sample of invalid ones are 0, or all invalid ones if feasible)
    
    # Actually, let's just check the required 54 entries equal 1.
    # The evaluator will check full validity, but we need to be reasonably sure.
    # Given small dimensions, let's check all valid entries are 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U[a][c] * V[b][c] * W[d][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Also check a few invalid entries to make sure they're 0
    # Total invalid: 9*18*18 - 54 = 2916 - 54 = 2862
    # Let's check all invalid entries since it's fast enough
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    
    for a in range(n*m):
        for b in range(m*p):
            for d in range(p*n):
                if (a, b, d) not in valid_set:
                    val = 0.0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[d][c]
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    n, m, p = 3, 3, 6
    
    # Try to find lower rank via random search with integer/half-integer entries
    # Rank bounds: rank >= max(n*m, m*p, p*n)/something? 
    # For matrix mult, rank >= 2*max(n,m,p) - something? 
    # Actually, known lower bound for M_{3,3,6} is at least 18 (from rank of M_{3,3,3} >= 9, tensor product gives 9*2=18)
    # Upper bound: 54 (schoolbook)
    
    # Try ranks from 18 upwards
    for rank in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 5000  # per rank
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Random initialization with small integers/half-integers
            # Use values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            U = [[float(np.random.choice(vals)) for _ in range(rank)] for _ in range(n*m)]
            V = [[float(np.random.choice(vals)) for _ in range(rank)] for _ in range(m*p)]
            W = [[float(np.random.choice(vals)) for _ in range(rank)] for _ in range(p*n)]
            
            # Quick check: for each valid (i,j,k), the sum should be 1
            # We can do a fast check
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        d = k * n + i
                        val = 0.0
                        for c in range(rank):
                            val += U[a][c] * V[b][c] * W[d][c]
                        if abs(val - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Full verification
                if _verify(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank; try lower ranks
            continue
        else:
            # Didn't find at this rank; try next (higher) rank
            # Actually we're going upward, so if we fail at rank, try rank+1
            pass
    
    return best
# EVOLVE-BLOCK-END
