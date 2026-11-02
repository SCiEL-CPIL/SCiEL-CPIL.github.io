"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
import random
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
    
    # Start with schoolbook
    U0, V0, W0, r0 = schoolbook(n, m, p)
    best = (U0, V0, W0, r0)
    
    # Try to reduce rank by finding linear dependencies
    # Represent each rank-1 component as a vector in R^(nm+mp+pn)
    # A component c has entries: U[:,c], V[:,c], W[:,c]
    # Two components are "equivalent" if they can be combined
    
    # Strategy: Try to find subsets of components that can be merged
    # For rank reduction, we look for linear relations
    
    # Alternative approach: Try structured low-rank decompositions
    # For n=2, we can try rank = 2*m*p = 50 (no improvement) 
    # or rank = m*p = 25 with special structure
    
    # Let's try a parameterized family for rank 25:
    # Component indexed by (j, k)
    # U[i*m+j', (j,k)] = delta(j,j') * a[i,k]
    # V[j'*p+k', (j,k)] = delta(j,j') * delta(k,k') * b[j,k]
    # W[k'*n+i', (j,k)] = delta(k,k') * d[i',j]
    
    # This gives: sum_{jc,kc} delta(j,jc)*a[i,kc] * delta(j2,jc)*delta(k2,kc)*b[jc,kc] * delta(k3,kc)*d[i3,jc]
    # = a[i,k2]*b[j,k2]*d[i3,j] if j=j2 and k2=k3, else 0
    
    # Need a[i,k]*b[j,k]*d[i3,j] = 1 when i=i3, 0 when i!=i3
    # This requires d[i,j]*d[i3,j] = 0 for i!=i3, and a[i,k]*b[j,k]*d[i,j] = 1
    # From d[i,j]*d[i3,j]=0: for each j, at most one of d[0,j], d[1,j] is nonzero
    # Let S0 = {j: d[0,j] != 0}, S1 = {j: d[1,j] != 0}, S0 ∪ S1 = {0..4}, S0 ∩ S1 = ∅
    
    # For i=0: a[0,k]*b[j,k]*d[0,j] = 1 for j in S0
    # For i=1: a[1,k]*b[j,k]*d[1,j] = 1 for j in S1
    
    # This is underdetermined. Let's try a specific assignment.
    # Let S0 = {0,1,2}, S1 = {3,4}
    
    # Try to solve numerically for rank 25
    # Variables: a[2,5], b[5,5], d[2,5] with constraints
    # This is a system of equations. Let's use a different approach.
    
    # Actually, let's try a more systematic search:
    # Try ranks from 25 to 50 and see if we can find a valid decomposition
    
    # For each candidate rank r, try random initialization + local search
    # But this is expensive. Let's try specific structured families first.
    
    # Family 1: rank = m*p = 25
    # Try the parameterization above with specific assignments
    
    # Direct sparse algebraic search for rank-25 decomposition
    # For each column j, assign to row 0 or row 1 (d[0,j]*d[1,j]=0)
    # Then solve: a[i,k]*b[j,k]*d[i,j] = 1 for valid (i,j) pairs
    
    found = False
    for mask in range(32):
        if time.time() > deadline - 10:
            break
        # S0 = bits set (columns assigned to row 0), S1 = bits not set (columns assigned to row 1)
        S0 = [j for j in range(m) if (mask >> j) & 1]
        S1 = [j for j in range(m) if not (mask >> j) & 1]
        
        if len(S0) == 0 or len(S1) == 0:
            continue
        
        # For this partition, we need:
        # For j in S0: a[0,k]*b[j,k]*d[0,j] = 1 for all k
        # For j in S1: a[1,k]*b[j,k]*d[1,j] = 1 for all k
        # And d[0,j]=0 for j in S1, d[1,j]=0 for j in S0
        
        # Try d[0,j] = 1 for j in S0, d[1,j] = 1 for j in S1
        # Then: a[0,k]*b[j,k] = 1 for j in S0, all k
        #       a[1,k]*b[j,k] = 1 for j in S1, all k
        
        # For consistency: b[j,k] must be the same function of k for all j in S0
        # and for all j in S1. So a[0,k] = a[1,k] = a[k] for all k
        # Then b[j,k] = 1/a[k] for all j
        
        # Set a[k] = 1 for all k, then b[j,k] = 1 for all j,k
        # This gives a valid decomposition!
        
        # Build U, V, W for rank 25
        r = 25
        U = [[0.0] * r for _ in range(nm)]
        V = [[0.0] * r for _ in range(mp)]
        W = [[0.0] * r for _ in range(pn)]
        
        for j in range(m):
            for k in range(p):
                c = j * p + k
                # U[i*m+j', c] = delta(j,j') * a[i,k]
                for i in range(n):
                    U[i*m+j][c] = 1.0  # a[i,k] = 1
                # V[j'*p+k', c] = delta(j,j') * delta(k,k') * b[j,k]
                V[j*p+k][c] = 1.0  # b[j,k] = 1
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
    
    # If rank 25 didn't work, try rank 26 with one extra component
    if not found and time.time() < deadline - 5:
        # Try adding one extra rank-1 component to break symmetry
        # This allows more flexible assignments
        r = 26
        U = [[0.0] * r for _ in range(nm)]
        V = [[0.0] * r for _ in range(mp)]
        W = [[0.0] * r for _ in range(pn)]
        
        # Use the same structure for first 25 components
        for mask in range(32):
            S0 = [j for j in range(m) if (mask >> j) & 1]
            S1 = [j for j in range(m) if not (mask >> j) & 1]
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
            
            # Add one extra component to handle any residual
            c = 25
            # Try to fix any off-diagonal terms with this extra component
            # For now, leave it zero and check if the base structure works
            if verify(U, V, W, n, m, p):
                found = True
                best = (U, V, W, r)
                break
    
    if not found and time.time() < deadline - 5:
        # Try a completely different approach: use numpy to solve a linear system
        # for a rank-25 decomposition with more general coefficients
        # This is expensive but might find a solution if the simple structure fails
        
        # For now, fall back to schoolbook
        pass
    
    return {"U": best[0], "V": best[1], "W": best[2]}
# EVOLVE-BLOCK-END
