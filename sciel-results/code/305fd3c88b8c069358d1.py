"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _schoolbook():
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k' else 0
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
    
    # Check diagonal: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: T[a,b,c] = 0 for all other (a,b,c)
    # We only need to check (a,b,c) where a=i*m+j, b=j2*p+k2, c=k2*n+i2
    # for some i,j,j2,k2,i2, and (i,j,k) != (i2,j2,k2)
    checked = set()
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k2 in range(p):
                    for i2 in range(n):
                        # a = i*m+j, b = j2*p+k2, c = k2*n+i2
                        # This is a valid (a,b,c) triple
                        # T should be 1 if i=i2 and j=j2 and there exists k with b=j*p+k and c=k*n+i
                        # i.e., if i=i2 and j=j2, then b must be j*p+k for some k, and c=k*n+i
                        a = i * m + j
                        b = j2 * p + k2
                        c = k2 * n + i2
                        if (a, b, c) in checked:
                            continue
                        checked.add((a, b, c))
                        
                        # Determine if T[a,b,c] should be 1 or 0
                        # T[a,b,c] = 1 iff there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
                        # a = i*m+j => i = a//m, j = a%m
                        # b = j2*p+k2 => this is already j2, k2
                        # c = k2*n+i2 => this is already k2, i2
                        # So T[a,b,c] = 1 iff i == i2 and j == j2 and the k from b matches the k from c
                        # k from b: k2 (since b = j2*p+k2)
                        # k from c: k2 (since c = k2*n+i2)
                        # So T[a,b,c] = 1 iff i == i2 and j == j2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val - expected) > 1e-9:
                            return False
    
    return True

def construct():
    start = time.time()
    deadline = start + 115  # 115 second limit to be safe
    
    # Start with schoolbook
    result = _schoolbook()
    
    return result
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
