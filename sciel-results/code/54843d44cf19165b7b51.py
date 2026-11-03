import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Split 2x3x6 into two 2x3x3 blocks (k=0,1,2 and k=3,4,5)
    # Find a low-rank decomposition for 2x3x3, then use 2 copies
    # This gives rank = 2 * (rank of 2x3x3 decomposition)
    
    # Try to find rank-r decomposition for 2x3x3
    # Schoolbook for 2x3x3 is rank 18
    # Try ranks from 12 to 17 (if found, full rank = 2*r)
    
    sub_n, sub_m, sub_p = 2, 3, 3
    sub_schoolbook_rank = sub_n * sub_m * sub_p  # 18
    
    best_sub_rank = None
    best_sub_U = None
    best_sub_V = None
    best_sub_W = None
    
    # Try to find a low-rank decomposition for 2x3x3
    # Search from low ranks upward
    for target_rank in range(12, sub_schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 30 if target_rank <= 14 else 15
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random half-integer matrices for U, V, W
            # U: (sub_n * sub_m) x target_rank = 6 x target_rank
            # V: (sub_m * sub_p) x target_rank = 9 x target_rank
            # W: (sub_p * sub_n) x target_rank = 6 x target_rank
            
            # Use small integers for better chance of exact solution
            max_val = 3
            U_rand = [[random.randint(-max_val, max_val) for _ in range(target_rank)] for _ in range(sub_n * sub_m)]
            V_rand = [[random.randint(-max_val, max_val) for _ in range(target_rank)] for _ in range(sub_m * sub_p)]
            W_rand = [[random.randint(-max_val, max_val) for _ in range(target_rank)] for _ in range(sub_p * sub_n)]
            
            # Check if this satisfies the tensor equation
            # For all i, j, k: sum_c U[i*sub_m+j, c] * V[j*sub_p+k, c] * W[k*sub_n+i, c] = 1
            valid = True
            for i in range(sub_n):
                for j in range(sub_m):
                    for k in range(sub_p):
                        s = 0
                        for c in range(target_rank):
                            s += U_rand[i * sub_m + j][c] * V_rand[j * sub_p + k][c] * W_rand[k * sub_n + i][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_sub_rank = target_rank
                best_sub_U = U_rand
                best_sub_V = V_rand
                best_sub_W = W_rand
                found = True
                break
        
        if found:
            break
    
    # If we found a sub-decomposition, construct the full decomposition
    if best_sub_rank is not None:
        full_rank = 2 * best_sub_rank
        if full_rank < best_rank:
            best_rank = full_rank
            U_full = [[0] * full_rank for _ in range(n * m)]
            V_full = [[0] * full_rank for _ in range(m * p)]
            W_full = [[0] * full_rank for _ in range(p * n)]
            
            # Block 1: k=0,1,2 (offset 0)
            # Block 2: k=3,4,5 (offset best_sub_rank)
            
            for rep in range(2):
                offset = rep * best_sub_rank
                k_offset = rep * sub_p  # 0 or 3
                
                for i in range(sub_n):
                    for j in range(sub_m):
                        for c in range(best_sub_rank):
                            U_full[i * sub_m + j][offset + c] = best_sub_U[i * sub_m + j][c]
                
                for j in range(sub_m):
                    for k in range(sub_p):
                        for c in range(best_sub_rank):
                            V_full[j * p + (k_offset + k)][offset + c] = best_sub_V[j * sub_p + k][c]
                
                for k in range(sub_p):
                    for i in range(sub_n):
                        for c in range(best_sub_rank):
                            W_full[(k_offset + k) * n + i][offset + c] = best_sub_W[k * sub_n + i][c]
            
            best_result = {"U": U_full, "V": V_full, "W": W_full}
    
    # If we didn't find a sub-decomposition, try a direct search for 2x3x6
    if best_rank == n * m * p and time.time() < deadline - 5:
        # Try a few random attempts for the full problem
        for target_rank in range(24, 36):
            if time.time() > deadline - 5:
                break
            
            max_attempts = 10
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                max_val = 3
                U_rand = [[random.randint(-max_val, max_val) for _ in range(target_rank)] for _ in range(n * m)]
                V_rand = [[random.randint(-max_val, max_val) for _ in range(target_rank)] for _ in range(m * p)]
                W_rand = [[random.randint(-max_val, max_val) for _ in range(target_rank)] for _ in range(p * n)]
                
                # Check validity
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += U_rand[i * m + j][c] * V_rand[j * p + k][c] * W_rand[k * n + i][c]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_rank = target_rank
                    best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                    break
            if best_rank < n * m * p:
                break
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
