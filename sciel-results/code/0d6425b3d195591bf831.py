"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if s != 1:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of target_rank by searching over integer/half-integer entries.
    Uses a structured approach: start from schoolbook and try to eliminate one tensor at a time.
    """
    start_rank = len(U[0])
    if target_rank >= start_rank:
        return U, V, W
    
    # Strategy: try to find a linear dependency among the rank-1 tensors.
    # For each column r to eliminate, try to express its contribution as a combination of others.
    # This means finding scalars alpha_r, beta_r, gamma_r for r != target such that:
    # For all (i,j,k): U[i*m+j, r_target] * V[j*p+k, r_target] * W[k*n+i, r_target]
    #   = sum_{r' != r_target} delta_r' * U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
    
    # This is hard in general. Instead, try a different approach:
    # Randomly initialize a lower-rank decomposition and use local search.
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to find a rank-35 decomposition by checking if any single term is redundant
    # For each term t, check if removing it and adjusting others can still give the correct tensor.
    
    for t in range(start_rank):
        if time.time() > deadline:
            break
        
        # Try to remove term t by finding adjustments to other terms
        # This requires solving a system of equations, which is complex.
        # Instead, use a heuristic: if two terms have the same "pattern" in one matrix,
        # they might be combinable.
        
        # Check if term t can be expressed as a combination of terms with the same U pattern
        u_pattern = tuple(U[i][t] for i in range(n*m))
        
        # Find other terms with the same U pattern
        same_u = [r for r in range(start_rank) if r != t and 
                  all(U[i][r] == U[i][t] for i in range(n*m))]
        
        if same_u:
            # Try to combine term t with one of the same-U-pattern terms
            for s in same_u:
                # If U patterns are the same, we can potentially combine V and W
                # Check if V[s] and V[t] and W[s] and W[t] allow combination
                v_diff = [V[j*p+k][t] - V[j*p+k][s] for j in range(m) for k in range(p)]
                w_diff = [W[k*n+i][t] - W[k*n+i][s] for k in range(p) for i in range(n)]
                
                # If the differences are proportional, we can merge
                # This is a simplified check
                if all(v == 0 for v in v_diff) and all(w == 0 for w in w_diff):
                    # Terms are identical, remove one
                    new_rank = start_rank - 1
                    new_U = [[0]*new_rank for _ in range(n*m)]
                    new_V = [[0]*new_rank for _ in range(m*p)]
                    new_W = [[0]*new_rank for _ in range(p*n)]
                    
                    col = 0
                    for r in range(start_rank):
                        if r == t:
                            continue
                        for i in range(n*m):
                            new_U[i][col] = U[i][r]
                        for idx in range(m*p):
                            new_V[idx][col] = V[idx][r]
                        for idx in range(p*n):
                            new_W[idx][col] = W[idx][r]
                        col += 1
                    
                    if _verify(new_U, new_V, new_W, n, m, p):
                        return new_U, new_V, new_W
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        pass  # Should not happen
    
    # Try to reduce rank by 1
    U, V, W = _try_rank_reduction(U, V, W, n, m, p, len(U[0]) - 1, deadline)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
