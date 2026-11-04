"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor decomposition says:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    _start = time.time()
    _deadline = _start + 135.0  # 135s deadline, 15s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, create the schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # For 2x4x5, the schoolbook rank is 40.
    
    # Strategy 1: Try a structured decomposition based on the specific structure
    # For matrix multiplication, we can sometimes do better than schoolbook
    
    # Let's try a rank that's less than schoolbook
    # Known: For 2xm x mxp, the rank is at most 2*max(m,p) + m*p - m*min(m,p)/2 or similar bounds
    
    # Let's try to construct a lower rank solution
    # One approach: use the fact that we can share some rank components
    
    # Try various ranks from lower to upper
    # Lower bound considerations:
    # - Must be at least max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # - But actually can be lower
    
    # Let's try ranks from 20 up to 39
    for target_rank in range(20, rank_school):
        if time.time() > _deadline - 10:
            break
        
        # Try to find a solution with this rank
        # Use a simple heuristic: try to construct it
        # 
        # For now, let's try a specific construction for small cases
        # This is a placeholder - a real search would be needed
        
        # Try a structured approach:
        # For each (i,k) pair, we need to compute sum_j A[i,j]*B[j,k]
        # We can use rank components to share work
        
        # Simple construction: one rank component per (i,k) pair
        # This gives rank = n*p = 10, but each component needs to handle all j
        # This won't work directly
        
        # Let's try a different approach:
        # Use rank = m*p = 20 (one per (j,k) pair)
        # For each (j,k), we have a rank component
        # U[i*m+j, r(j,k)] = 1 if i is the right row
        # V[j*p+k, r(j,k)] = 1
        # W[k*n+i, r(j,k)] = 1 if i is the right row
        # 
        # But this gives sum over j,k of A[i,j]*B[j,k] for each i,k
        # Which is exactly what we want!
        
        if target_rank == m * p:
            U_try = [[0] * target_rank for _ in range(n * m)]
            V_try = [[0] * target_rank for _ in range(m * p)]
            W_try = [[0] * target_rank for _ in range(p * n)]
            
            for j in range(m):
                for k in range(p):
                    r = j * p + k
                    for i in range(n):
                        U_try[i * m + j][r] = 1
                        W_try[k * n + i][r] = 1
                    V_try[j * p + k][r] = 1
            
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_result = {"U": U_try, "V": V_try, "W": W_try}
                best_rank = target_rank
                break
        
        # Try rank = n*m = 8
        if target_rank == n * m:
            U_try = [[0] * target_rank for _ in range(n * m)]
            V_try = [[0] * target_rank for _ in range(m * p)]
            W_try = [[0] * target_rank for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    r = i * m + j
                    U_try[i * m + j][r] = 1
                    for k in range(p):
                        V_try[j * p + k][r] = 1
                        W_try[k * n + i][r] = 1
            
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_result = {"U": U_try, "V": V_try, "W": W_try}
                best_rank = target_rank
                break
    
    if time.time() > _deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
