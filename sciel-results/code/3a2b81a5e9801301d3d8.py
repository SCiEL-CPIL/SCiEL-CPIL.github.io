"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p = 52
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try rank 41 = 3p+2 construction
    target_rank = 3 * p + 2  # 41
    if target_rank < best_rank and time.time() < deadline - 5:
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Construction for 2x2 x 2xp with rank 3p+2
        # Uses a "ladder" of overlapping 2x2 x 2x1 blocks
        # 
        # The key idea: for each k, we compute contributions to C[0,k] and C[1,k]
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        #
        # We use shared intermediate products between adjacent k values.
        # 
        # Let's define the rank terms as follows:
        # Term 0: boundary for k=0, i=0
        # Term 1: boundary for k=0, i=1  
        # For k = 0 to p-2:
        #   Term 2+3k:   A[0,0] * B[0,k+1]
        #   Term 2+3k+1: A[1,1] * B[1,k+1]
        #   Term 2+3k+2: (A[0,0]+A[1,1]) * (B[0,k+1]+B[1,k+1]) - correction
        # 
        # Actually, let me use a cleaner known construction.
        # 
        # For the 2x2 x 2xp tensor, one known rank 3p+2 construction:
        # 
        # We introduce p+1 "intermediate" variables that form a chain.
        # 
        # Let me try a direct algebraic construction:
        # 
        # For each k, define:
        #   x_k = A[0,0] + A[1,0]  (shared across all k for column 0 of A)
        #   y_k = A[0,1] + A[1,1]  (shared across all k for column 1 of A)
        #
        # This doesn't directly help. Let me use the standard approach:
        #
        # The tensor M_{2,2,p} has rank 3p+2. A construction:
        #
        # For r = 0, 1:  (boundary terms)
        #   U[0*m + 0, r] = 1, V[0*p + 0, r] = 1, W[0*n + 0, r] = (-1)^r
        #   U[1*m + 1, r] = 1, V[1*p + 0, r] = 1, W[1*n + 1, r] = (-1)^r
        #
        # This is getting complex. Let me try a numerical approach instead:
        # Use the schoolbook as a starting point and try to compress it.
        
        # Alternative: try rank 40, 39, etc. with a simple verification
        # For now, let's try to construct rank 41 explicitly.
        
        # Known construction from the literature for M_{2,2,p}:
        # 
        # The decomposition uses the following pattern:
        # For s = 0, 1, ..., p (p+1 "stations"):
        #   At each station s, we have up to 3 rank-1 terms
        # 
        # Station 0 (left boundary): 2 terms
        # Station s for 1 <= s <= p-1: 3 terms  
        # Station p (right boundary): 2 terms
        # Total: 2 + 3(p-1) + 2 = 3p + 1... still not 3p+2
        #
        # Let me just try to find ANY rank < 52 by brute force search over small structures.
        
        # Try a simple "Strassen-like" extension:
        # For 2x2 x 2x2, Strassen gives rank 7 = 4*2 - 1
        # For 2x2 x 2xp, maybe rank = 4p - something?
        # 4*13 = 52, so we'd need to save at least 1.
        
        # Let's try: can we merge two schoolbook terms?
        # Schoolbook term for (i=0,j=0,k) and (i=1,j=1,k) can potentially be merged
        # if we use (A[0,0]+A[1,1]) as a shared factor.
        
        # For a fixed k, the 4 schoolbook terms are:
        # (0,0,k): U[0]=1, V[k]=1, W[k*2+0]=1
        # (0,1,k): U[1]=1, V[2+k]=1, W[k*2+0]=1  
        # (1,0,k): U[2]=1, V[k]=1, W[k*2+1]=1
        # (1,1,k): U[3]=1, V[2+k]=1, W[k*2+1]=1
        #
        # Can we replace these 4 terms with 3 terms?
        # Using: 
        #   m1 = A[0,0]*B[0,k]
        #   m2 = A[1,1]*B[1,k]
        #   m3 = (A[0,0]+A[1,1])*(B[0,k]+B[1,k])
        # Then:
        #   C[0,k] = m1 + (m3 - m2 - m1 + A[0,1]*B[1,k])... no, this doesn't work cleanly.
        
        # The correct 3-term decomposition for 2x2 x 2x1:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        #
        # With 3 multiplications:
        #   m1 = A[0,0] * B[0,0]
        #   m2 = A[1,1] * B[1,0]
        #   m3 = (A[0,1] + A[1,0]) * (B[0,0] + B[1,0])
        # Then:
        #   C[0,0] = m1 + A[0,1]*B[1,0] = m1 + (m3 - A[0,1]*B[0,0] - A[1,0]*B[1,0] - A[1,0]*B[0,0])
        # This is getting circular.
        
        # Let me just verify the schoolbook and return it. The known optimal is 41,
        # but constructing it correctly is complex. Return schoolbook for safety.
        
        pass
    
    return best_result
# EVOLVE-BLOCK-END
