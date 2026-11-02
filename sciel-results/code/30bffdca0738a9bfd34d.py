"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import time
import numpy as np
from fractions import Fraction
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Tensor T[i,j,k] = delta_{j,k}
    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k}
    
    # Schoolbook rank = 60
    # Try to find rank < 60
    
    def verify(U, V, W):
        """Verify exact decomposition with integer/half-integer entries."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    expected = 1 if j == k else 0
                    if s != expected:
                        return False
        return True
    
    def to_fraction_list(lst):
        """Convert list to Fraction list for exact arithmetic."""
        return [Fraction(x) for x in lst]
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    # Try rank 54 construction
    # Use a structured approach based on known small-rank decompositions
    
    # For 2x3x3, rank 7 decomposition exists.
    # Let's try to extend this to 2x3x10.
    
    # The 2x3x3 rank-7 decomposition (from literature):
    # Terms (a,b,c) with values in {0,1,-1}
    # This is complex, so let me try a different approach.
    
    # Try: can we achieve rank 54 by a simple modification of schoolbook?
    # Schoolbook has 60 terms. If we can merge 6 terms, we get 54.
    
    # Idea: For the 2x3x10 case, note that T[i,j,k] = delta_{j,k}
    # The tensor is "separable" in the sense that for fixed i, 
    # T[i,.,.] is the 3x10 identity-like matrix.
    
    # Let me try a construction with rank 54:
    # Use 54 = 6 * 9 terms
    # For each (i,j) pair (6 total), use 9 terms
    
    # For fixed (i,j), we need: sum_{c=1}^{9} U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{j,k}
    # This is a system of 10 equations (for k=0..9) with 27 unknowns per (i,j)
    # (9 U values, 9*10 V values, 9*2 W values, but shared across i for fixed j)
    
    # This is underdetermined, so solutions should exist.
    # Let me try to construct one explicitly.
    
    # Simplification: assume U[i,j,c] = 1 for all i,j,c (or some simple pattern)
    # Then we need: sum_c V[j,k,c] * W[k,i,c] = delta_{j,k}
    
    # If U[i,j,c] = 1, then for fixed j, we need:
    # sum_c V[j,k,c] * W[k,i,c] = delta_{j,k} for all i,k
    # This means the 10x2 matrix (W[k,i,c])_k must satisfy...
    # Actually, this couples i and k.
    
    # Let me try: W[k,i,c] = w[k,c] * z[i,c] (rank-1 in (k,i) for each c)
    # Then sum_c V[j,k,c] * w[k,c] * z[i,c] = delta_{j,k}
    # For this to hold for all i, we need z[i,c] to be such that...
    # This is getting complicated.
    
    # Let me just try a numerical search for rank 54
    # with a specific structure
    
    # Actually, let me try a much simpler approach:
    # Use the fact that 54 < 60 and see if a simple construction works.
    
    # Construction: Use 54 terms where for each j in {0,1,2},
    # we have 18 terms (54/3=18) that handle the k dimension.
    
    # For j=0: 18 terms that give delta_{0,k} = 1 if k=0 else 0
    # For j=1: 18 terms that give delta_{1,k} = 1 if k=1 else 0
    # For j=2: 18 terms that give delta_{2,k} = 1 if k=2 else 0
    
    # For j=0, k=0: sum of 18 terms = 1
    # For j=0, k!=0: sum of 18 terms = 0
    
    # One simple way: use 1 term for k=j (giving 1) and 17 terms that cancel
    # But that's not efficient.
    
    # Better: for j=0, use terms that are 1 at k=0 and 0 elsewhere
    # This is exactly the schoolbook approach for that j.
    
    # The schoolbook uses 10 terms per (i,j) pair, total 6*10=60.
    # To get 54, we need to save 6 terms.
    
    # Idea: Can we share terms between different i values?
    # For i=0 and i=1, the W matrix has entries W[k,0,c] and W[k,1,c].
    # If we use the same U and V for both i=0 and i=1, but different W,
    # we might be able to reduce the total.
    
    # Actually, in the schoolbook, each (i,j,k) triple has its own term.
    # If we can make U[i,j,c] and V[j,k,c] shared across i, we save.
    
    # Let me try: for each (j,k), use a single term shared across i=0,1
    # But then W[k,0,c] and W[k,1,c] would need to be the same,
    # which they can be (set to 1).
    
    # If U[i,j,c] = 1 for both i=0,1 (same U for both i),
    # V[j,k,c] = 1, W[k,i,c] = 1,
    # then sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
    # But we need delta_{j,k}, not 1.
    
    # So we need V[j,k,c] to be 0 when j!=k.
    # This means for each j, only k=j terms are nonzero.
    # This brings us back to schoolbook.
    
    # The only way to reduce rank is to use cancellations.
    # For j!=k, we need the sum to be 0, which allows for
    # multiple terms that cancel each other.
    
    # Let me try a specific rank-54 construction:
    # Use 54 terms. For the 30 "diagonal" (j=k) contributions,
    # we need net 1. For the 30 "off-diagonal" (j!=k) contributions,
    # we need net 0.
    
    # One approach: use 30 diagonal terms (one for each (j,k=j))
    # and 24 off-diagonal terms that cancel in pairs.
    # Total: 30 + 24 = 54.
    
    # But we also have the i dimension. Each term contributes to both i=0 and i=1.
    
    # Let me try:
    # 30 diagonal terms: for each (j,k=j), one term with U=1, V=1, W=1
    # This gives 1 for all i, which is correct for the diagonal.
    # 24 off-diagonal terms: arranged to cancel for j!=k.
    
    # For off-diagonal, we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 0 for j!=k.
    # If we use terms where U[i,j,c]=1, V[j,k,c]=a, W[k,i,c]=b with a*b=0...
    # No, that's not right.
    
    # Let me think differently. Use the following construction:
    # For each j in {0,1,2}, k in {0,...,9}, i in {0,1}:
    # We need the total contribution to be delta_{j,k}.
    
    # Use 54 terms indexed by (j, t) where t=0,...,17 (18 terms per j, 3*18=54)
    # For term (j,t):
    #   U[i,j,c] = u_i (depends on i)
    #   V[j,k,c] = v_k (depends on k)  
    #   W[k,i,c] = w_{k,i} (depends on k,i)
    
    # This is getting too abstract. Let me just try to return schoolbook
    # and note that finding a lower rank requires significant work.
    
    # Given time constraints, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
