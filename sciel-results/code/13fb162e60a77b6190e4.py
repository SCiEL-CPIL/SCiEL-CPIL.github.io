"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify exact decomposition."""
        r = len(U[0])
        # Check diagonal
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return U, V, W, rank
    
    # Start with schoolbook as fallback
    U0, V0, W0, r0 = schoolbook(n, m, p)
    best = (U0, V0, W0, r0)
    
    # Strategy: Try to find a rank-25 decomposition using the structure
    # For n=2, we can partition columns of V between the two rows of W
    
    # The key insight: for rank = m*p = 25, we use one component per (j,k) pair
    # Component c = j*p + k has:
    #   U[i*m+j', c] = delta(j,j') * a[i,k]
    #   V[j'*p+k', c] = delta(j,j') * delta(k,k') * b[j,k]  
    #   W[k'*n+i', c] = delta(k,k') * d[i',j]
    
    # This gives sum over components:
    #   For output (i3, j2, k2): sum_{j,k} delta(j,j2)*a[i3,k] * delta(j,j2)*delta(k,k2)*b[j2,k2] * delta(k,k2)*d[i3,j2]
    #   = a[i3,k2]*b[j2,k2]*d[i3,j2] if j2 and k2 are valid, else 0
    #   Wait, this isn't quite right. Let me reconsider.
    
    # Actually the product is:
    #   U[i*m+j, c] * V[j2*p+k2, c] * W[k2*n+i3, c]
    #   = delta(j,jc)*a[i,kc] * delta(j2,jc)*delta(k2,kc)*b[jc,kc] * delta(k2,kc)*d[i3,jc]
    #   where c = jc*p + kc
    #   = a[i,k2]*b[j2,k2]*d[i3,j2] if j=j2 and k2=k2, else 0
    #   = a[i,k2]*b[j2,k2]*d[i3,j2] when j=j2, else 0
    
    # So for the diagonal term (i=j2=k2=i3): we need a[i,k]*b[i,k]*d[i,i] = 1... no wait
    # The diagonal is (i, j, k) -> output (k, i) in W convention W[k*n+i]
    # So U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = delta(j,jc)*a[i,kc] * delta(j,jc)*delta(k,kc)*b[jc,kc] * delta(k,kc)*d[i,jc]
    # = a[i,k]*b[j,k]*d[i,j] when j=jc and k=kc
    
    # For off-diagonal (i, j, k) with (j2, k2, i3) != (j, k, i):
    # If j2 != j: delta(j,j2)=0, so term is 0
    # If k2 != k: delta(k,k2)=0, so term is 0  
    # If i3 != i: we need d[i,j]*d[i3,j] = 0 for all j
    
    # So the conditions are:
    # 1. a[i,k]*b[j,k]*d[i,j] = 1 for all i,j,k
    # 2. d[i,j]*d[i3,j] = 0 for i != i3, all j
    
    # From condition 2: for each j, at most one of d[0,j], d[1,j] is nonzero
    # Let S0 = {j : d[0,j] != 0}, S1 = {j : d[1,j] != 0}, S0 ∪ S1 = {0,...,m-1}
    
    # From condition 1:
    # For j in S0: a[0,k]*b[j,k]*d[0,j] = 1 for all k
    # For j in S1: a[1,k]*b[j,k]*d[1,j] = 1 for all k
    # For j in S0: a[1,k]*b[j,k]*d[1,j] = 0 (since d[1,j]=0)
    # For j in S1: a[0,k]*b[j,k]*d[0,j] = 0 (since d[0,j]=0)
    
    # This is consistent if we choose:
    # d[0,j] = 1 for j in S0, d[1,j] = 1 for j in S1
    # a[0,k] = 1 for all k, a[1,k] = 1 for all k
    # b[j,k] = 1 for all j,k
    
    # Then condition 1 becomes: 1*1*1 = 1 ✓
    
    # So any non-trivial partition works!
    
    # Try all partitions of {0,...,m-1} into S0 and S1
    found = False
    for mask in range(1 << m):
        if time.time() > deadline - 10:
            break
        
        S0 = [j for j in range(m) if (mask >> j) & 1]
        S1 = [j for j in range(m) if not ((mask >> j) & 1)]
        
        if len(S0) == 0 or len(S1) == 0:
            continue
        
        r = m * p  # = 25
        U = [[0.0] * r for _ in range(nm)]
        V = [[0.0] * r for _ in range(mp)]
        W = [[0.0] * r for _ in range(pn)]
        
        for j in range(m):
            for k in range(p):
                c = j * p + k
                # U[i*m+j', c] = delta(j,j') * a[i,k] = delta(j,j') * 1
                for i in range(n):
                    U[i*m+j][c] = 1.0
                # V[j'*p+k', c] = delta(j,j') * delta(k,k') * b[j,k] = delta(j,j') * delta(k,k') * 1
                V[j*p+k][c] = 1.0
                # W[k'*n+i', c] = delta(k,k') * d[i',j]
                for i3 in range(n):
                    if (i3 == 0 and j in S0) or (i3 == 1 and j in S1):
                        W[k*n+i3][c] = 1.0
                    else:
                        W[k*n+i3][c] = 0.0
        
        if verify(U, V, W, n, m, p):
            found = True
            best = (U, V, W, r)
            break
    
    # If rank 25 didn't work, try rank 26
    if not found and time.time() < deadline - 5:
        # Try a more general approach with rank 26
        # Add one extra component to handle any issues
        r = 26
        U = [[0.0] * r for _ in range(nm)]
        V = [[0.0] * r for _ in range(mp)]
        W = [[0.0] * r for _ in range(pn)]
        
        for mask in range(1 << m):
            if time.time() > deadline - 5:
                break
            S0 = [j for j in range(m) if (mask >> j) & 1]
            S1 = [j for j in range(m) if not ((mask >> j) & 1)]
            
            if len(S0) == 0 or len(S1) == 0:
                continue
            
            for j in range(m):
                for k in range(p):
                    c = j * p + k
                    for i in range(n):
                        U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    for i3 in range(n):
                        if (i3 == 0 and j in S0) or (i3 == 1 and j in S1):
                            W[k*n+i3][c] = 1.0
            
            # Leave the 26th component as zero for now
            if verify(U, V, W, n, m, p):
                found = True
                best = (U, V, W, r)
                break
    
    if not found and time.time() < deadline - 5:
        # Try random search for lower rank decompositions
        # This is expensive but might find something
        
        # For now, fall back to schoolbook
        pass
    
    return {"U": best[0], "V": best[1], "W": best[2]}
# EVOLVE-BLOCK-END
