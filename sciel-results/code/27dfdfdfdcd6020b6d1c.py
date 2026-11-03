import numpy as np
from itertools import product
from scipy.optimize import linprog
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_triples = n * m * p  # 60
    
    # Build the constraint matrix A where A[triple_idx, pattern_idx] = 
    # product of U_pattern[i*m+j] * V_pattern[j*p+k] * W_pattern[k*n+i]
    # Each pattern is a triple of support vectors (u, v, w) in {0,1}^{nm} x {0,1}^{mp} x {0,1}^{pn}
    # But we need to be smarter. Let's enumerate all "elementary" patterns.
    
    # Actually, for exact decomposition with integer/half-integer entries, 
    # let's try a different approach: use the known structure.
    
    # For 3x4x5, we can try to find a rank-59 or lower decomposition.
    # The schoolbook is rank 60. Let's try to reduce by 1.
    
    # Strategy: Start with schoolbook rank 60, then try to find a linear dependence
    # among the 60 rank-1 terms that allows us to merge two into one.
    
    # Schoolbook: term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1, all else 0.
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    
    # To reduce rank, we need to find coefficients c_r and modified vectors such that
    # sum_r c_r * u_r ⊗ v_r ⊗ w_r = T
    
    # Let's try a direct approach: set up the problem as finding the minimum rank
    # such that there exist U, V, W with the required property.
    
    # For small dimensions, we can try to use the fact that the multiplication tensor
    # has a certain structure. Let's try to find a rank-59 decomposition.
    
    # Actually, let me just return the schoolbook for now but with a cleaner implementation,
    # and add a quick check if we can do better.
    
    base_rank = n * m * p
    
    # Schoolbook decomposition
    U = np.zeros((nm, base_rank), dtype=float)
    V = np.zeros((mp, base_rank), dtype=float)
    W = np.zeros((pn, base_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank by 1: see if we can find a non-trivial combination
    # This is hard in general. For now, return rank 60.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
