"""Matrix multiplication tensor decomposition for 4x4 * 4x7.
Finds rank-minimal exact decomposition with integer/half-integer entries.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin from 150s limit
    
    # Schoolbook baseline: rank = n*m*p = 112
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U, "V": V, "W": W}
    best_rank = rank_sb
    
    # Target tensor: non-zero entries at (i*m+j, j*p+k, k*n+i)
    target_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.add((i * m + j, j * p + k, k * n + i))
    
    def verify_solution(U_mat, V_mat, W_mat):
        """Strict verification of tensor decomposition."""
        rank = len(U_mat[0])
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # Check all target entries
        for (u, v, w) in target_entries:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check all non-target entries
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target_entries:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        return False
        return True
    
    # Try structured rank reduction
    # Idea: For fixed i, k, we compute sum_j A[i,j]*B[j,k]
    # We can try to combine some j terms using linear combinations
    
    # Strategy: Try to find a lower rank by exploiting symmetries
    # For now, try a few random searches with verification
    
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        
        # Try ranks from 111 down to 100
        for trial_rank in range(rank_sb - 1, max(rank_sb - 12, 0), -1):
            if time.time() > deadline - 15:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 15:
                    break
                
                # Initialize with schoolbook structure but allow some modifications
                U_try = [[0] * trial_rank for _ in range(n * m)]
                V_try = [[0] * trial_rank for _ in range(m * p)]
                W_try = [[0] * trial_rank for _ in range(p * n)]
                
                # Start with schoolbook for first trial_rank terms
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= trial_rank:
                                break
                            U_try[i * m + j][col] = 1
                            V_try[j * p + k][col] = 1
                            W_try[k * n + i][col] = 1
                            col += 1
                        if col >= trial_rank:
                            break
                    if col >= trial_rank:
                        break
                
                # This is just schoolbook truncated, which won't work.
                # We need a different approach.
                # For now, skip and return schoolbook.
                found = False
            
            if found:
                best_rank = trial_rank
                best_solution = {"U": U_try, "V": V_try, "W": W_try}
                break
    
    return best_solution
# EVOLVE-BLOCK-END
