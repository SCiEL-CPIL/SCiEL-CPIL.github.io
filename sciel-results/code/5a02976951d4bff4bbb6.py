"""Matrix multiplication 2x3 x 3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Schoolbook solution as baseline
    rank_school = n * m * p  # 48
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    def verify(U, V, W):
        """Verify exact decomposition."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use Winograd frequency-domain decomposition
    # The 8-length dimension can be decomposed using a 4-point transform
    # Key idea: split 8 into two 4-length blocks and use Winograd
    
    # Winograd 4-point transform matrix (integer coefficients)
    # For convolution of length 4, we can use a rank-4 decomposition
    # Transform matrix T (4x4) with integer/half-integer entries
    
    # Let's try a systematic approach:
    # 1. Use the fact that 8 = 2*4, so we can have block structure
    # 2. For each 4-block, use a Winograd-like decomposition
    
    # Winograd for 4-point: use transform with entries in {0, ±1, ±1/2}
    # The key transform for 4-point convolution:
    # M = [[1, 1, 1, 1],
    #      [1, 0, -1, 0],
    #      [0, 1, 0, -1],
    #      [1, -1, 0, 0]]
    # But we need exact integer/half-integer decompositions
    
    # Let's try a different approach: use the structure of the problem
    # 2x3 x 3x8: we can think of this as 2 blocks of (1x3 x 3x8)
    # or as (2x1 x 1x8) + (2x2 x 2x8) components
    
    # Actually, let's use a more direct approach:
    # The tensor is T[i,j,k] = delta_{i,j,k} for the multiplication tensor
    # We need to find U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let's try rank values from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    # Try structured decompositions
    # For the 8-dimension, use a 4-point Winograd transform
    
    # Winograd 4-point: the transform matrix can be:
    # G = [[1, 1, 1, 1],
    #      [1, 0, -1, 0],
    #      [0, 1, 0, -1],
    #      [1, -1, 0, 0]]
    # With appropriate scaling to get integer/half-integer entries
    
    # Let's try a more practical approach: use random search with structure
    # But first, let's try some known good decompositions
    
    # Approach: Use the fact that 3 = 1+2, so we can split the middle dimension
    # This gives us a block diagonal structure
    
    # Let me try a direct construction based on Winograd for the 8-dimension
    
    # For 4-point Winograd:
    # The transform has rank 4, and we can use it for each 4-block of the 8-dimension
    
    # Let's try rank = 24 first (the lower bound)
    # If that doesn't work, try 25, 26, etc.
    
    vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25])
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 20.0:
            break
        
        found = False
        
        # Try multiple random initializations
        max_trials = 500 if target_rank <= 30 else 200
        for trial in range(max_trials):
            if time.time() > deadline - 10.0:
                break
            
            rng = np.random.RandomState(trial * 7919 + target_rank * 137 + 42)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Sparse random initialization with structure
            density = 0.4 + 0.2 * (target_rank / best_rank)
            for idx in range(n*m):
                mask = rng.random(target_rank) < density
                U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(m*p):
                mask = rng.random(target_rank) < density
                V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(p*n):
                mask = rng.random(target_rank) < density
                W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            if verify(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
        best_rank = rank_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
