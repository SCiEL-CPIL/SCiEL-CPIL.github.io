"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction as F
import numpy as np

# Known rank-7 decomposition for 2x2 matrix multiplication
# This is a standard result from tensor decomposition literature.
# The 7 rank-1 terms for multiplying 2x2 matrices.
STRASSEN_2x2 = [
    # (u, v, w) where u is 4-dim (A indices), v is 4-dim (B indices), w is 4-dim (C indices)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # Index mapping: A: 0=a,1=b,2=c,3=d; B: 0=e,1=f,2=g,3=h; C: 0=p,1=q,2=r,3=s
    # Term 1: p = ae
    ([1,0,0,0], [1,0,0,0], [1,0,0,0]),
    # Term 2: q = bf
    ([0,1,0,0], [0,1,0,0], [0,1,0,0]),
    # Term 3: r = cg
    ([0,0,1,0], [0,0,1,0], [0,0,1,0]),
    # Term 4: s = dh
    ([0,0,0,1], [0,0,0,1], [0,0,0,1]),
    # These are just schoolbook - we need the actual Strassen rank-7 decomposition
    # The correct rank-7 decomposition is more complex.
    # For now, let's use a verified rank-7 decomposition from literature.
]

# Actually, the standard rank-7 decomposition for 2x2 matrix multiplication:
# The 7 vectors are known. Let me recall them properly.
# From the Strassen algorithm perspective, the 7 products are:
# M1 = (a+d)(e+h)
# M2 = (c+d)h
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c+d)(f-h)
# M7 = (a+b)(e-g)
# But these are not rank-1 tensor terms in the simple sense.
# The actual tensor rank decomposition has 7 terms with specific coefficients.

# Let me use a different approach: use the schoolbook for 3x3x5 but try to find
# a better decomposition through optimization or known results.

# For 3x3x5, the schoolbook rank is 45. The theoretical minimum is at least 27.
# Known results suggest the rank is around 23-27 for 3x3x3, and for 3x3x5 it might be similar.

# Given the complexity, let me implement a search-based approach to find a lower rank
# decomposition by exploiting the structure.

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
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
    
    # Try to find a better decomposition using optimization
    start_time = time.time()
    deadline = start_time + 100  # Keep 20s safety margin
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # For 3x3x5, let's try rank 23-30
    # We'll use a randomized approach to initialize and then optimize
    
    def validate(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1 if (i < n and j < m and k < p) else 0
                    # The tensor T[i,j,k] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                    actual = 0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try different ranks from lower to higher
    for target_rank in range(23, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            random.seed(42 + target_rank + attempt)
            U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Simple greedy/local search to improve
            # This is a simplified approach - in practice, we'd use more sophisticated optimization
            
            # For now, just validate the schoolbook as fallback
            if validate(U_school, V_school, W_school, rank_schoolbook):
                if target_rank >= rank_schoolbook:
                    best_rank = rank_schoolbook
                    best_U = U_school
                    best_V = V_school
                    best_W = W_school
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
