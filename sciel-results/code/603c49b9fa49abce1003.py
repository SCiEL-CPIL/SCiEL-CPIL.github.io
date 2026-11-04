"""Tensor decomposition for 2x3x4 matrix multiplication via integer/rational search."""
import numpy as np
from itertools import product
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    total_terms = n * m * p  # 24
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need rank r such that there exist U(nm x r), V(mp x r), W(pn x r) with
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Known lower bound: max(nm, mp, pn) = max(6, 12, 8) = 12
    # But the actual rank for 2x3x4 is known to be around 19-24.
    # Let's try to find a decomposition with rank as low as possible.
    
    # Strategy: Try ranks from the lower bound upward, using a randomized
    # approach with rational entries to find exact decompositions.
    
    # First, let's try a structured approach.
    # The schoolbook gives rank 24. We want to do better.
    
    # Approach: Use the fact that we can combine terms.
    # For a 2x3x4 matrix multiplication, the tensor has 24 non-zero entries.
    # We want to find r vectors (u_c, v_c, w_c) in the three spaces such that
    # their tensor sum equals the target tensor.
    
    # Let's try rank from 12 upward, using random initialization with small integers
    # and then checking if we can solve the linear system for one factor.
    
    # Actually, a cleaner approach: fix U and V, then W is determined by solving
    # a linear system. But the system is nonlinear (product of three).
    
    # Better approach: Use the "rank 1" decomposition idea.
    # Each rank-1 term contributes U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] to T[i,j,k].
    # We need sum over c of these = 1 for all (i,j,k).
    
    # Let's try a different strategy: use known constructions.
    # For 2x2x2, rank is 7 (Strassen). For 2x3x4, we can try to build on similar ideas.
    
    # Let me try a direct search with small integer entries.
    # We'll use rank r and try to find integer solutions.
    
    # Key insight: We can parameterize the problem. For a given rank r,
    # we have r * (nm + mp + pn) = r * (6 + 12 + 8) = 26r variables.
    # We have 24 equations (one for each (i,j,k) pair).
    # So for r >= 2, we have more variables than equations, suggesting solutions exist.
    
    # Let's try to find solutions with small integer entries.
    
    # Strategy: For each rank r starting from 12, try random small integer initializations
    # and use Newton's method or direct solving.
    
    # Actually, let's use a simpler approach: 
    # For rank r, set up the problem as: find U, V, W with entries in {-1, 0, 1, 1/2, ...}
    # such that the tensor equation holds.
    
    # Let me try a constructive approach for specific ranks.
    
    # For rank 24 (schoolbook), we already have a solution.
    # Let's see if we can do better.
    
    # Try rank 19-23 with random search.
    
    best_result = None
    best_rank = 24
    
    # Start with the schoolbook solution as fallback
    U_school = [[0]*24 for _ in range(6)]
    V_school = [[0]*24 for _ in range(12)]
    W_school = [[0]*24 for _ in range(8)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 24
    
    # Now try to find lower rank solutions
    # Use random search with small rational entries
    
    np.random.seed(42)
    
    for r in range(12, 24):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_attempts = 5000 if r >= 18 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers
            U = np.zeros((nm, r), dtype=float)
            V = np.zeros((mp, r), dtype=float)
            W = np.zeros((pn, r), dtype=float)
            
            # Fill with random values from a small set
            vals = [-1, -0.5, 0, 0.5, 1]
            for c in range(r):
                U[:, c] = np.array([np.random.choice(vals) for _ in range(nm)])
                V[:, c] = np.array([np.random.choice(vals) for _ in range(mp)])
                W[:, c] = np.array([np.random.choice(vals) for _ in range(pn)])
            
            # Check if this satisfies the tensor equation
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                # Convert to list format
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    # If we found a solution with rank < 24, return it
    # Otherwise, return the schoolbook solution
    
    if best_rank < 24:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
