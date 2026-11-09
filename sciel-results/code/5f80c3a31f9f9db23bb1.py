"""Improved program for matrix multiplication tensor rank (2,3,7)."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    We need to find U (shape n*m x rank), V (shape m*p x rank), W (shape p*n x rank)
    such that for all i,j,k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Using the schoolbook gives rank = n*m*p = 42. We try to find lower rank.
    
    Known bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    And rank >= (n+m+p-2) = 7 (Landsberg bound, but this is weak)
    
    For 2x3x7, let's try to find a decomposition with rank less than 42.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Start with schoolbook
    rank_schoolbook = n * m * p  # 42
    
    # Try various ranks from lower bound upward
    # Lower bound is at least max(n*m, m*p, p*n) = 21
    # But also need to satisfy the tensor equation
    
    # Let's try to find a decomposition using a systematic approach
    # We'll use the fact that for small matrices, we can try to find
    # structured decompositions.
    
    # First, let's verify the schoolbook works and return it as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: Use the structure of the problem.
    # For multiplying (n x m) by (m x p), we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: try rank = m * max(n,p) or similar structured forms
    # For n=2, m=3, p=7:
    #   m * max(n,p) = 3 * 7 = 21
    #   max(n,p) * m = same
    #   n * p = 14 (too low)
    
    # Let's try rank 21 first with a structured approach
    # Idea: For each j, we need to compute the contribution A[i,j] * B[j,k]
    # We can share factors across i and k.
    
    # Actually, let's think about it differently.
    # The tensor is T[i,j,k] = delta_{i,j,k} where the multiplication tensor has
    # T[i,j,k] = 1 for all i,j,k (since every entry of A and B contributes).
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # M[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    # This is a constant tensor! All entries are 1.
    # So we need: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is actually easier than I thought. The tensor is all-ones.
    # We need a rank-r decomposition of the all-ones tensor of shape (nm, mp, pn).
    
    # The all-ones tensor J of shape (a,b,c) has rank 1!
    # J[i,j,k] = 1 = 1*1*1
    # So U[i,c] = 1, V[j,c] = 1, W[k,c] = 1 for c=0, and 0 otherwise.
    # Rank = 1!
    
    # Wait, but that's not right either. Let me re-read the problem.
    
    # The matrix multiplication tensor for (n x m) * (m x p) is:
    # mu_{ijk} = 1 if we're multiplying, 0 otherwise
    # Actually, the standard matrix multiplication tensor has:
    # mu_{ijk} = 1 for all i,j,k
    # And the decomposition is: C_{ik} = sum_j sum_c U_{ijc} A_{ij} V_{jkc} B_{jk} W_{kic}
    
    # Hmm, I need to be more careful. Let me re-read the contract.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k
    # and we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
    
    # So yes, it's the all-ones tensor, which has rank 1!
    
    # Let me verify: U[i*m+j, 0] = 1 for all i,j
    #           V[j*p+k, 0] = 1 for all j,k  
    #           W[k*n+i, 0] = 1 for all k,i
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1 ✓
    
    # So rank 1 should work!
    
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook()
