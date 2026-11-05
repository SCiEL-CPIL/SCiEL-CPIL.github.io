"""
Program for finding a low-rank exact tensor decomposition of the 2x3 x 3x5 matrix multiplication tensor.

The task is to find factor matrices U (6 x r), V (15 x r), W (10 x r) such that
for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is the only term
    and 0 for all other triples.

The schoolbook decomposition uses r = n*m*p = 30. We try to find a lower rank.

Strategy:
- Use an iterative local search / coordinate descent to try to reduce rank below 30.
- Start from the schoolbook solution at rank 30, then try to merge or eliminate terms.
- Alternatively, try to construct a decomposition at lower rank using randomized initialization
  and verify exactness.

Given the difficulty of finding a provably lower rank for 2x3x5 (known bounds suggest
the rank is between 21 and 30), we will attempt a search but fall back to schoolbook.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]
n, m, p = DIMENSIONS

def _validate(U, V, W):
    """Check that U, V, W form an exact decomposition of the matmul tensor."""
    r = len(U[0])
    # Build the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Target: T[i,j,k] = 1 if this is the "identity" term, but actually for matrix multiplication
    # the tensor has T[i,j,k] = 1 for all (i,j,k) because C_ik = sum_j A_ij * B_jk
    # Wait, the matmul tensor has T_{i,j,k} = 1 for all valid i,j,k.
    # Let me re-read the task.
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The condition is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is because C_ik = sum_j A_ij B_jk, so each term A_ij * B_jk contributes to C_ik.
    
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_numpy(U, V, W):
    """Vectorized validation using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    r = U_np.shape[1]
    
    # U has shape (n*m, r), V (m*p, r), W (p*n, r)
    # For each (i,j,k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We can do this efficiently by reshaping
    # Let's just loop over i, j, k since n*m*p = 30 is small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]  # shape (r,)
                v_val = V_np[j*p+k, :]  # shape (r,)
                w_val = W_np[k*n+i, :]  # shape (r,)
                s = np.dot(np.dot(u_val, v_val), w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 100.0  # Leave margin for the 120s limit
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook()
    best_rank = n * m * p  # 30
    
    # Try to find lower rank decompositions
    # Strategy 1: Try rank 29, 28, ... down to some minimum
    # For each rank, try randomized initialization + local search
    
    min_rank_to_try = 21  # Known lower bound
    
    # Try descending from best_rank - 1 down to min_rank_to_try
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        if _try_find_decomposition(target_rank, deadline):
            # Found a decomposition at this rank
            # Try to go lower
            continue
        # If we couldn't find one at this rank, maybe try a lower rank
        # Actually, let's try a few attempts at each rank
    
    # Return the best we found
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_find_decomposition(target_rank, deadline):
    """Try to find a decomposition of the matmul tensor at the given rank.
    
    Uses randomized initialization followed by iterative refinement.
    Returns True if a valid decomposition is found, False otherwise.
    """
    attempts = 10
    for attempt in range(attempts):
        if time.time() > deadline - 5.0:
            return False
        
        # Random initialization with small values
        U = [[random.uniform(-0.5, 0.5) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.uniform(-0.5, 0.5) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.uniform(-0.5, 0.5) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Iterative refinement: try to reduce the error
        # The residual for each (i,j,k) is:
        # r[i,j,k] = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We want all residuals to be zero.
        
        # Simple approach: gradient-free local search
        # Try perturbing individual entries and accept if it reduces total squared error
        
        best_err = _total_squared_error(U, V, W)
        
        max_iterations = 200
        for it in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            if best_err < 1e-12:
                return True
            
            improved = False
            # Try perturbing a random entry in U, V, or W
            for _ in range(20):  # Try 20 random perturbations per iteration
                mat = random.choice([0, 1, 2])  # 0=U, 1=V, 2=W
                if mat == 0:
                    M = U
                    rows = n * m
                elif mat == 1:
                    M = V
                    rows = m * p
                else:
                    M = W
                    rows = p * n
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, target_rank - 1)
                old_val = M[row][col]
                delta = random.uniform(-0.1, 0.1)
                M[row][col] = old_val + delta
                
                new_err = _total_squared_error(U, V, W)
                if new_err < best_err:
                    best_err = new_err
                    improved = True
                else:
                    M[row][col] = old_val
            
            if not improved and it > 50:
                break
        
        if best_err < 1e-12:
            return True
    
    return False

def _total_squared_error(U, V, W):
    """Compute the sum of squared residuals over all (i,j,k)."""
    r = len(U[0])
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                total += (1.0 - s) ** 2
    return total
