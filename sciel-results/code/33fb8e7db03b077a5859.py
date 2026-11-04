import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # The trivial decomposition has rank n*m*p = 50
    # We try to find a decomposition with rank 49 by solving a linear system
    # where we fix one tensor entry to be 1 and others to be 0,
    # and use the structure of rank-1 terms.
    
    # For rank r, we have r terms. Each term t has:
    # u_t in R^(n*m), v_t in R^(m*p), w_t in R^(p*n)
    # The tensor T_{i,j,k} = sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i]
    
    # This is nonlinear. However, we can use the following trick:
    # Fix the W matrix to be a standard basis or some simple pattern,
    # then the system becomes linear in U and V.
    
    # Let's try: fix W to have rank r with specific structure.
    # A common approach: set w_t to be the standard basis vectors in R^(p*n)
    # if r >= p*n, but r=49 < 50, so we can't do that directly.
    
    # Alternative: use the fact that we only need to match 50 entries.
    # Let's try to find a solution with r=49 by:
    # 1. Fixing one entry of the tensor to 0 (say T[0,0,0]=0)
    # 2. Setting up a linear system for the remaining 49 entries
    #    with r=49 terms, where we fix U and V to be standard basis-like
    
    # Actually, let's try a different approach:
    # Use the known fact that for 2x5x5, the rank might be 49.
    # We can try to construct it explicitly.
    
    # Let's try to find a rank-49 decomposition by solving a linear system.
    # We'll fix the W matrix to be 49x50 with specific values,
    # then solve for U and V.
    
    # Here's a concrete attempt:
    # Let the 49 terms correspond to all (i,j,k) except (0,0,0).
    # For each term t corresponding to (i_t, j_t, k_t), set:
    # u_t[i*m+j] = 1 if (i,j) == (i_t, j_t), else 0
    # v_t[j*p+k] = 1 if (j,k) == (j_t, k_t), else 0
    # w_t[k*n+i] = 1 if (k,i) == (k_t, i_t), else 0
    
    # Then T[i,j,k] = sum_{t: i_t=i, j_t=j, k_t=k} 1*1*1
    # This gives T[i,j,k] = 1 for all (i,j,k) except (0,0,0) which is 0.
    # But we need T[i,j,k] = 1 for all (i,j,k).
    
    # So this doesn't work directly. We need a different construction.
    
    # Let's just return the trivial decomposition with rank 50.
    # This guarantees a valid solution.
    
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
                
    return {"U": U, "V": V, "W": W}
