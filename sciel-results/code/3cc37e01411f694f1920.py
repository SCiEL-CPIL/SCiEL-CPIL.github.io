"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
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
    
    # Strategy: Try rank 25 with structured parameterization exploiting n=2
    # Use the fact that for n=2, we can set up a linear system for the W factors
    # given U and V, then search for sparse U, V.
    
    # Parameterization for rank r:
    # U has shape (n*m, r) = (10, r)
    # V has shape (m*p, r) = (25, r)  
    # W has shape (p*n, r) = (10, r)
    
    # Key insight: Fix U and V, then W is determined by linear equations.
    # The tensor equation is: for each (i,j,j2,k,k2,i3):
    #   sum_c U[i*m+j,c] * V[j2*p+k,c] * W[k2*n+i3,c] = delta(i,i3)*delta(j,j2)*delta(k,k2)
    
    # For fixed U,V, this is a linear system in W.
    # Let's try to find sparse U,V such that the resulting W is also sparse/structured.
    
    # Try rank 25 with a specific structure:
    # Let U[i*m+j, c] where c = j0*p + k0 (c in 0..24)
    # Structure: U[i*m+j, j0*p+k0] = delta(j, j0) * A[i, k0]
    # V[j*p+k, j0*p+k0] = delta(j, j0) * delta(k, k0)
    # W[k*n+i, j0*p+k0] = delta(k, k0) * B[i, j0]
    
    # Then: sum_c U[i*m+j,c]*V[j2*p+k2,c]*W[k3*n+i3,c]
    # = sum_{j0,k0} delta(j,j0)*A[i,k0] * delta(j2,j0)*delta(k2,k0) * delta(k3,k0)*B[i3,j0]
    # = A[i,k2]*B[i3,j] if j=j2 and k2=k3, else 0
    
    # Diagonal condition: A[i,k]*B[i,j] = 1 for all i,j,k
    # This forces A[i,k] = 1 and B[i,j] = 1, giving the schoolbook with rank 25? 
    # No wait, that gives rank 25 but let's check: each component (j0,k0) has
    # U[i*m+j0, j0*p+k0] = A[i,k0], V[j0*p+k0, j0*p+k0] = 1, W[k0*n+i, j0*p+k0] = B[i,j0]
    # The number of non-zero components is 25, so rank = 25.
    # But does A[i,k]*B[i,j] = 1 for all i,j,k have a solution?
    # A[i,k] must be independent of j, B[i,j] must be independent of k.
    # A[i,k]*B[i,j] = 1 for all j,k means A[i,k] = 1/B[i,j] for all j, so B[i,j] is constant in j.
    # Let B[i,j] = b_i for all j. Then A[i,k] = 1/b_i for all k.
    # So A[i,k] = 1/b_i and B[i,j] = b_i.
    # This works! Let b_0 = 1, b_1 = 1. Then A[i,k] = 1, B[i,j] = 1.
    
    # Wait, but this gives the same as schoolbook essentially. Let me verify:
    # For i=0, j=0, k=0: component (j0=0, k0=0): U[0*5+0, 0*5+0]=A[0,0]=1, V[0*5+0,0*5+0]=1, W[0*2+0,0*5+0]=B[0,0]=1. Product=1. Good.
    # For i=0, j=0, k=0 with j2=1: sum over c where j=0=j0, k2=0=k0: A[0,0]*B[i3,0]*delta(k3,0). Only i3=0 gives B[0,0]=1, but j2=1 != j0=0 so V[1*5+0, 0*5+0]=delta(1,0)*delta(0,0)=0. Good.
    
    # So rank 25 works! Let's construct it.
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # b[i] values
    b = [1, 1]  # for i=0,1
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            for i in range(n):
                U[i * m + j0][c] = 1 / b[i]
                W[k0 * n + i][c] = b[i]
            V[j0 * p + k0][c] = 1
    
    # Verify
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (unlikely to be reached)
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
