"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 145.0  # 5s safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # For n=3, m=4, p=7, the matmul tensor has entries T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # The actual rank is likely higher. Let's try to find a good decomposition.
    
    # Strategy: Use the fact that we can exploit structure
    # For matrix multiplication, we can sometimes find decompositions using
    # block structures or other algebraic identities
    
    # Let's try a systematic search with small integer values
    # Focus on finding decompositions with rank < 84
    
    np.random.seed(12345)
    
    # Try various ranks, starting from lower values
    # The minimum possible is 28, but realistic values are probably in the 40-70 range
    
    # For each target rank, try random integer matrices and check
    min_possible = max(n*m, m*p, p*n)  # = 28
    
    # Try ranks from min_possible up to rank_school - 1
    # But be strategic: try a few promising ranges
    ranks_to_try = list(range(min_possible, rank_school))
    
    # Sort by likelihood of success (try middle values first, then expand)
    mid = (min_possible + rank_school) // 2
    ranks_to_try.sort(key=lambda r: abs(r - mid))
    
    found = False
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 20:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random attempts for this rank
        for attempt in range(100):
            if time.time() > DEADLINE - 10:
                break
                
            # Random integer matrices with small values
            # Use values from {-1, 0, 1} or {-2, -1, 0, 1, 2}
            max_val = 2 if target_rank < 50 else 3
            
            U = np.random.randint(-max_val, max_val + 1, size=(n*m, target_rank))
            V = np.random.randint(-max_val, max_val + 1, size=(m*p, target_rank))
            W = np.random.randint(-max_val, max_val + 1, size=(p*n, target_rank))
            
            # Check if this decomposition works
            # For speed, use numpy to compute all entries at once
            # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        # Compute sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
                        total = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        if total != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                found = True
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found:
            break
    
    # If we haven't found a good one, try half-integer values
    if not found and time.time() < DEADLINE - 30:
        for target_rank in ranks_to_try[:20]:  # Try first 20 ranks
            if time.time() > DEADLINE - 15:
                break
            if target_rank >= best_rank:
                continue
                
            for attempt in range(50):
                if time.time() > DEADLINE - 10:
                    break
                    
                # Half-integer values: use odd integers divided by 2
                U = (np.random.randint(-3, 4, size=(n*m, target_rank)) * 2 + 1)  # odd integers
                V = (np.random.randint(-3, 4, size=(m*p, target_rank)) * 2 + 1)
                W = (np.random.randint(-3, 4, size=(p*n, target_rank)) * 2 + 1)
                
                # When we divide by 2, the product is divided by 8
                # So we need sum_c (U/2)(V/2)(W/2) = 1
                # Which means sum_c U*V*W = 8
                
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            total = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                            if total != 8:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                
                if success:
                    found = True
                    best_rank = target_rank
                    # Convert to half-integers
                    U_frac = [[U[i][c] / 2 for c in range(target_rank)] for i in range(n*m)]
                    V_frac = [[V[i][c] / 2 for c in range(target_rank)] for i in range(m*p)]
                    W_frac = [[W[i][c] / 2 for c in range(target_rank)] for i in range(p*n)]
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    break
            
            if found:
                break
    
    return best_result
# EVOLVE-BLOCK-END
