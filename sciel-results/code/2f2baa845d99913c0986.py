"""Rewritten program for Matrix multiplication 2,2,11 rank minimization."""
import numpy as np
from scipy.optimize import linprog
import itertools
from collections import defaultdict

DIMENSIONS = [2, 2, 11]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x11 matrix.
    
    Key insight: For each (j,k) pair, we can create a single rank-1 term that handles
    both i=0 and i=1 contributions simultaneously.
    
    For fixed j and k:
    - C[0,k] gets contribution A[0,j]*B[j,k]
    - C[1,k] gets contribution A[1,j]*B[j,k]
    
    We can use one rank-1 term per (j,k) pair where:
    - U[0*m+j, r] = 1, U[1*m+j, r] = 1 (same j, different i)
    - V[j*p+k, r] = 1
    - W[k*n+0, r] = 1, W[k*n+1, r] = 1
    
    This gives rank = m*p = 2*11 = 22, which is half the schoolbook rank of 44.
    
    Verification: For output entry C[k*n+i]:
    Sum over all rank terms r: U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k*n+i, r]
    
    Each term r corresponds to a unique (j_r, k_r) pair.
    U[i*m+j_r, r] = 1 only if j_r is the j-index for this term
    V[j_r*p+k_r, r] = 1 only if (j_r, k_r) matches
    W[k*n+i, r] = 1 only if k_r = k and i is in {0,1}
    
    So for a given (i,k), the sum becomes:
    Sum over j: U[i*m+j, r_{j,k}] * V[j*p+k, r_{j,k}] * W[k*n+i, r_{j,k}]
    = Sum over j: 1 * 1 * 1 = m = 2
    
    Wait, that gives 2, not 1. I need to adjust.
    
    Let me reconsider. The correct approach:
    
    For each (j,k) pair, create a term with:
    - U[i*m+j, r] = delta_{i,0} for i=0, delta_{i,1} for i=1? No, U must be constant.
    
    Actually, the issue is that U[i*m+j, r] must be the same for all k, but we want
    different behavior for different k.
    
    Let me think again. The standard schoolbook uses 44 terms: one for each (i,j,k).
    
    Can we combine terms for different i? For fixed (j,k):
    Term for (i=0, j, k): U[0*m+j, r1]=1, V[j*p+k, r1]=1, W[k*n+0, r1]=1
    Term for (i=1, j, k): U[1*m+j, r2]=1, V[j*p+k, r2]=1, W[k*n+1, r2]=1
    
    These two terms have the same V entry but different U and W entries.
    
    If we could use a single term r for both, we'd need:
    U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
    U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] = 1
    
    And for other (i',k') combinations, the product should be 0.
    
    This is tricky because U[0*m+j, r] and U[1*m+j, r] are fixed values, and
    W[k*n+0, r] and W[k*n+1, r] are fixed values.
    
    Let me try a different approach: use the structure of the problem.
    
    For n=2, m=2, p=11, the tensor has 44 non-zero entries.
    
    One known result: for matrix multiplication of (n x m) * (m x p), the rank is
    at most n*m*p, but can be lower.
    
    Let me try to find a decomposition with rank 22 by using the following construction:
    
    For each j in {0,1} and k in {0,...,10}, define term r_{j,k}:
    - U[i*m+j, r_{j,k}] = 1 for all i (i.e., U[0*m+j, r]=1 and U[1*m+j, r]=1)
    - V[j*p+k, r_{j,k}] = 1
    - W[k*n+i, r_{j,k}] = 1 for all i (i.e., W[k*n+0, r]=1 and W[k*n+1, r]=1)
    
    Then for output C[k*n+i]:
    Sum_r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k*n+i, r]
    
    = Sum over (j_r, k_r) where j_r=j and k_r=k: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    = Sum over j: 1 * 1 * 1 = 2
    
    This gives 2, not 1. So this construction doesn't work directly.
    
    What if I scale? If I set U[i*m+j, r] = 1/2, then the sum would be 2*(1/2)*1*1 = 1.
    But the problem requires integer/half-integer entries, so 1/2 is allowed!
    
    Let me verify: with U[i*m+j, r_{j,k}] = 1/2, V[j*p+k, r_{j,k}] = 1, W[k*n+i, r_{j,k}] = 1:
    
    For C[k*n+i]:
    Sum over all r: U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k*n+i, r]
    
    Only terms where j_r=j and k_r=k contribute (since V[j_r*p+k_r, r] is non-zero only for specific (j_r,k_r)).
    
    Wait, V[j_r*p+k_r, r] = 1 for the specific (j_r, k_r) associated with term r.
    So V[j*p+k, r] = 1 only if (j, k) = (j_r, k_r) for that term.
    
    So for a given output (i,k), the sum is over all terms r where k_r = k:
    Sum over j: U[i*m+j, r_{j,k}] * V[j*p+k, r_{j,k}] * W[k*n+i, r_{j,k}]
    = Sum over j: (1/2) * 1 * 1 = 2 * (1/2) = 1
    
    This works! And for other (a,b,c) combinations where the tensor should be 0:
    
    Consider T[a,b,c] where a = i'*m+j', b = j''*p+k'', c = k'*n+i''
    with j' != j'' or k' != k or i' != i'' or other mismatches.
    
    The sum is over r: U[a, r] * V[b, r] * W[c, r]
    
    For a given r corresponding to (j_r, k_r):
    U[a, r] = 1/2 if a = i*m+j_r for some i, else 0
    V[b, r] = 1 if b = j_r*p+k_r, else 0
    W[c, r] = 1 if c = k_r*n+i for some i, else 0
    
    So the product is non-zero only if:
    - a = i1*m+j_r for some i1
    - b = j_r*p+k_r
    - c = k_r*n+i2 for some i2
    
    This means j' = j_r, k'' = k_r, k' = k_r, so k'' = k'.
    And i1 and i2 can be anything.
    
    But we need T[a,b,c] = 0 unless (a,b,c) corresponds to a valid multiplication entry.
    
    A valid entry has a = i*m+j, b = j*p+k, c = k*n+i (same i, j, k).
    
    In our sum, for terms where k_r = k'' = k', we get contributions from all i1, i2.
    But we only want the contribution when i1 = i2 = i (the same i).
    
    So the sum gives:
    Sum over i1, i2: U[i1*m+j_r, r] * V[j_r*p+k_r, r] * W[k_r*n+i2, r]
    = Sum over i1, i2: (1/2) * 1 * 1 = 4 * (1/2) = 2
    
    But we want this to be 0 for invalid (i1, i2) combinations!
    
    This construction doesn't work because it creates unwanted cross-terms.
    
    Let me try a different approach. What if I use different values for U depending on i?
    
    For term r_{j,k}:
    U[0*m+j, r] = 1
    U[1*m+j, r] = 0
    V[j*p+k, r] = 1
    W[k*n+0, r] = 1
    W[k*n+1, r] = 0
    
    And another term r'_{j,k}:
    U[0*m+j, r'] = 0
    U[1*m+j, r'] = 1
    V[j*p+k, r'] = 1
    W[k*n+0, r'] = 0
    W[k*n+1, r'] = 1
    
    This is just the schoolbook decomposition with 44 terms.
    
    Can I combine these two terms into one? If I use:
    U[0*m+j, r] = 1
    U[1*m+j, r] = 1
    V[j*p+k, r] = 1
    W[k*n+0, r] = 1
    W[k*n+1, r] = 1
    
    Then for valid entry (i,j,k):
    Contribution = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1 ✓
    
    For invalid entry (i1,j,k) with i1 != i2 (where c = k*n+i2):
    If i1 != i2, then W[k*n+i2, r] = 1, but we're computing T[i1*m+j, j*p+k, k*n+i2]
    Contribution = U[i1*m+j, r] * V[j*p+k, r] * W[k*n+i2, r] = 1*1*1 = 1 ✗ (should be 0)
    
    So this doesn't work.
    
    Let me try yet another approach. What if I use the following construction:
    
    For each j in {0,1}, I'll create p terms.
    For term r_{j,k}:
    U[i*m+j, r] = 1 for all i
    V[j*p+k, r] = 1
    W[k*n+i, r] = 1 for all i
    
    But as we saw, this creates cross-terms.
    
    What if I use different weights for different i in W?
    
    Let me try:
    U[i*m+j, r_{j,k}] = 1 for all i
    V[j*p+k, r_{j,k}] = 1
    W[k*n+0, r_{j,k}] = 1
    W[k*n+1, r_{j,k}] = 1
    
    And accept that this gives 2 for valid entries. Then I can scale by 1/2:
    U[i*m+j, r_{j,k}] = 1/2 for all i
    V[j*p+k, r_{j,k}] = 1
    W[k*n+i, r_{j,k}] = 1 for all i
    
    For valid entry (i,j,k): sum = 2 * (1/2)*1*1 = 1 ✓
    For invalid entry (i1,j,k) with i1 != i2: sum = (1/2)*1*1 = 1/2 ✗
    
    Still doesn't work.
    
    I think the fundamental issue is that we can't easily combine terms for different i without creating cross-terms.
    
    Let me just return the schoolbook decomposition with rank 44. It's correct and guaranteed to work.
    """
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
