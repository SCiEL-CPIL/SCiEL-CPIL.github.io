"""Matrix multiplication 3,4,5 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import sys
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook decomposition as fallback
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
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += float(U[i*m+j][r]) * float(V[j*p+k][r]) * float(W[k*n+i][r])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def to_integer_list(lst, tol=1e-6):
        """Convert float list to integer list, returning None if fails."""
        result = []
        for x in lst:
            if abs(x) < tol:
                result.append(0)
            else:
                rounded = round(x)
                if abs(rounded - x) < tol:
                    result.append(int(rounded))
                else:
                    return None
        return result
    
    # Try to find lower rank decompositions using structured approaches
    # Strategy: Use the linear structure by considering the middle index j
    
    # Approach 1: Try to find a decomposition where the factors have a specific structure
    # For example, assume U[i*m+j, r] = u_i[j, r], V[j*p+k, r] = v_j[k, r], W[k*n+i, r] = w_k[i, r]
    # This is already the general form, so we need a different structure.
    
    # Approach 2: Use a change of basis. If we can diagonalize the middle index coupling,
    # we might reduce rank. But for exact integer solutions, this is tricky.
    
    # Approach 3: Randomized search with verification for small rank reductions
    # Try ranks from 50 to 59
    
    if time.time() > deadline:
        return best_result
    
    random.seed(42)
    
    # Try a few candidate ranks
    candidate_ranks = [55, 50, 45, 40]
    candidate_ranks = [r for r in candidate_ranks if r < best_rank]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try random initialization with small integers
        attempts = 10
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W_rand = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_decomposition(U_rand, V_rand, W_rand, rank):
                # Convert to integers if possible
                U_int = to_integer_list([x for row in U_rand for x in row])
                V_int = to_integer_list([x for row in V_rand for x in row])
                W_int = to_integer_list([x for row in W_rand for x in row])
                
                if U_int is not None and V_int is not None and W_int is not None:
                    U_final = [U_int[i*rank:(i+1)*rank] for i in range(n*m)]
                    V_final = [V_int[i*rank:(i+1)*rank] for i in range(m*p)]
                    W_final = [W_int[i*rank:(i+1)*rank] for i in range(p*n)]
                    best_result = {"U": U_final, "V": V_final, "W": W_final}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
