"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    # Try to find lower rank decompositions
    # The tensor condition: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try structured decompositions
    # Idea: for each i, the row computation C[i,:] = A[i,:] * B
    # This is a (1 x m) * (m x p) multiplication, which has rank m schoolbook.
    # But we can share structure across i's.
    
    # Let's try: for each i, use a rank-m decomposition of the row multiplication
    # But share the V factors across i's if possible.
    
    # Actually, let me try a direct random search for small ranks
    # Start from rank = 48 and try to reduce
    
    # Alternative: use the fact that we can write the multiplication as
    # C = A @ B, and find a low-rank decomposition of this bilinear map
    
    # Let me try a systematic approach: fix the structure
    # For (2,4,6): 
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # We need: for each (i,j,k): sum_c U[i*4+j,c] * V[j*6+k,c] * W[k*2+i,c] = 1
    
    # Let's try rank = 24 (half of schoolbook) with a structured approach
    # Idea: pair up the j indices
    
    # Try various rank values from high to low
    for target_rank in range(47, 15, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition with this rank
        # Use a simple iterative approach: initialize randomly, then refine
        
        # Actually, let me try a more structured approach
        # For (2,4,6), note that 4 = 2*2, so maybe we can use a 2x2 block structure
        
        # Let me try: for each i (0,1), for each j_pair (0,1) where j = 2*j_pair + offset
        # This might not work directly...
        
        # Let me try a different structured approach:
        # Use the fact that we can factor the multiplication
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j (A[i,j] * B[j,k])
        
        # If we could write A[i,j] * B[j,k] as a sum of rank-1 terms in some way...
        
        # Actually, let me just try a random search with verification
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random rational-like values (use small integers)
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Random initialization
            for c in range(target_rank):
                # Assign random small integer values
                for idx in range(n * m):
                    U[idx, c] = random.randint(-2, 2)
                for idx in range(m * p):
                    V[idx, c] = random.randint(-2, 2)
                for idx in range(p * n):
                    W[idx, c] = random.randint(-2, 2)
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(target_rank))
                        if val != 1:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                # Convert to lists of integers
                U_list = [[int(U[idx, c]) for c in range(target_rank)] for idx in range(n*m)]
                V_list = [[int(V[idx, c]) for c in range(target_rank)] for idx in range(m*p)]
                W_list = [[int(W[idx, c]) for c in range(target_rank)] for idx in range(p*n)]
                best = {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
                break
        
        if best["rank"] < rank_school:
            break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
