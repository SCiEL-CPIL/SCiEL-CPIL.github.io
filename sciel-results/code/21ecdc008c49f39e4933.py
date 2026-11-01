"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    The schoolbook rank is 2*3*4 = 24.
    We attempt to find a lower rank decomposition, but if not found quickly,
    return the schoolbook decomposition.
    
    The tensor T has indices (i, j, k) where i in [0,2), j in [0,3), k in [0,4).
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (12, rank)
    W has shape (p*n, rank) = (8, rank) with W[k*n+i, r]
    
    The decomposition requires:
    For all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Since T[i,j,k] = 1 for all valid i,j,k, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook decomposition
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Return schoolbook as fallback
    result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a lower rank decomposition using random search
    # The known lower bounds for 2x3x4 matrix multiplication are not well-established
    # in simple cases, so we try a few random approaches
    
    start_time = time.time()
    deadline = 110  # Keep well under 120s limit
    
    # Try rank 23, 22, 21, 20 with random integer/half-integer entries
    # Use structured random search with better initialization
    for target_rank in range(schoolbook_rank - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random decompositions
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Generate random matrices with small integer/half-integer entries
            # Use a structured approach based on tensor decomposition theory
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))
            W_rand = np.zeros((p * n, target_rank))
            
            # Strategy: Start from schoolbook and try to merge terms
            # Each term in schoolbook corresponds to one (i,j,k) triple
            # We try to create terms that cover multiple triples
            
            # Initialize with random sparse structure
            for r in range(target_rank):
                # Each term should contribute to a small set of (i,j,k) triples
                # Use 2-3 triples per term on average
                num_terms = random.randint(1, 3)
                for _ in range(num_terms):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    # Use coefficients that help with cancellation
                    coeff = random.choice([1, -1, 0.5, -0.5, 2, -2])
                    U_rand[i * m + j, r] += coeff
                    V_rand[j * p + k, r] += coeff
                    W_rand[k * n + i, r] += coeff
            
            # Check if this decomposition works
            # For each (i,j,k), check if sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U_rand[i * m + j, r] * V_rand[j * p + k, r] * W_rand[k * n + i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                U_list = [[float(U_rand[i, r]) for r in range(target_rank)] for i in range(n * m)]
                V_list = [[float(V_rand[i, r]) for r in range(target_rank)] for i in range(m * p)]
                W_list = [[float(W_rand[i, r]) for r in range(target_rank)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return schoolbook
    return result
# EVOLVE-BLOCK-END
