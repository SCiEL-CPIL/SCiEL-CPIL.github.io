"""Program for matrix multiplication 2x3x6 tensor decomposition."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    The tensor T has entries T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k'
    and 0 otherwise.
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Start with schoolbook rank 36, then try to reduce
    # The tensor has 36 non-zero entries
    
    # Try rank 18: one term per (j,k) pair
    # For term (j,k): need U[i*m+j]*V[j*p+k]*W[k*n+i] to work for both i=0 and i=1
    # But U and W depend on i, so we can't have a single term work for both i values
    # unless U[0*m+j] = U[1*m+j] and W[k*n+0] = W[k*n+1], which would give wrong results
    
    # Let me try a systematic search for lower rank decompositions
    # using the structure of the problem
    
    # Key insight: The tensor is "block diagonal" in a sense - 
    # terms for different (j,k) pairs don't interact
    # So the rank is at least the sum of ranks of each (j,k) block
    # Each (j,k) block is a 2x1x2 tensor (for i and i' dimensions)
    # The rank of a 2x1x2 tensor with entries [[1,0],[0,1]] is 2
    # So total rank >= 18 * 2 = 36? No, that's not right.
    
    # Actually, let me reconsider. The tensor T(a,b,c) where:
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T(a,b,c) = 1 if i=i', j=j', k=k', else 0
    
    # For a rank-1 term: U[a]*V[b]*W[c]
    # This can only be non-zero when a,b,c are consistent
    # Specifically, if U is non-zero at position i*m+j, V at j'*p+k, W at k'*n+i'
    # then the product is non-zero only if these indices are "compatible"
    
    # The constraint is: for the product to contribute to T, we need
    # i=i', j=j', k=k'
    # So U must be supported on positions with a specific (i,j),
    # V on positions with (j',k) where j'=j,
    # W on positions with (k',i') where k'=k and i'=i
    
    # This means each rank-1 term can only contribute to ONE (i,j,k) triple
    # OR to multiple triples if the supports overlap in a compatible way
    
    # Wait, that's not quite right. A rank-1 term U*V*W has:
    # U supported on some set of (i,j) positions
    # V supported on some set of (j',k) positions  
    # W supported on some set of (k',i') positions
    # The product is non-zero at (a,b,c) = (i*m+j, j'*p+k, k'*n+i')
    # iff U[i*m+j]*V[j'*p+k]*W[k'*n+i'] != 0
    # And it contributes to T only if i=i', j=j', k=k'
    
    # So for a rank-1 term to contribute to T(i,j,k)=1, we need:
    # U[i*m+j] != 0, V[j*p+k] != 0, W[k*n+i] != 0
    # AND the product U[i*m+j]*V[j*p+k]*W[k*n+i] should equal 1 (or sum to 1 over terms)
    
    # But the term also contributes to other (i',j',k') triples if:
    # U[i'*m+j'] != 0, V[j'*p+k'] != 0, W[k'*n+i'] != 0
    # and i'=i, j'=j, k'=k (for it to be a valid T entry)
    # Actually no - it contributes to T(i',j',k') only if i'=i, j'=j, k'=k
    # Wait, I'm confusing myself.
    
    # Let me be very explicit. The tensor T has entries indexed by (a,b,c) where:
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T(a,b,c) = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    # T(a,b,c) = 0 otherwise
    
    # A rank-1 decomposition term is U[a]*V[b]*W[c]
    # The sum over all terms should equal T(a,b,c)
    
    # For a specific term to contribute to T(i*m+j, j*p+k, k*n+i) = 1,
    # we need U[i*m+j]*V[j*p+k]*W[k*n+i] to be part of the sum
    
    # But this same term also has values at other (a',b',c') positions:
    # U[a']*V[b']*W[c'] for all a',b',c'
    # Most of these will be 0 if U,V,W are sparse
    # But if U is non-zero at multiple positions, the term contributes to multiple T entries
    
    # The key constraint: if U is non-zero at (i1,j1) and (i2,j2),
    # V is non-zero at (j1,k1) and (j2,k2),
    # W is non-zero at (k1,i1) and (k2,i2),
    # then the term contributes to:
    # T(i1*m+j1, j1*p+k1, k1*n+i1) += U[i1*m+j1]*V[j1*p+k1]*W[k1*n+i1]
    # T(i2*m+j2, j2*p+k2, k2*n+i2) += U[i2*m+j2]*V[j2*p+k2]*W[k2*n+i2]
    # AND cross terms like:
    # T(i1*m+j1, j2*p+k1, k1*n+i2) += U[i1*m+j1]*V[j2*p+k1]*W[k1*n+i2]
    # But this is 0 in T unless i1=i2, j1=j2, k1=k1 (which requires i1=i2 and j1=j2)
    
    # So cross terms are problematic! They contribute to T entries that should be 0.
    
    # This means: for a rank-1 term to be valid (not contribute to wrong T entries),
    # we need: if U is non-zero at (i1,j1) and (i2,j2), and V is non-zero at (j1,k1) and (j2,k2),
    # then we must have i1=i2 and j1=j2 (otherwise cross terms appear)
    # Similarly for other combinations.
    
    # In other words, the supports of U, V, W must be "consistent" in a strong sense.
    # Specifically, the set of (i,j) pairs where U is non-zero, combined with
    # the set of (j,k) pairs where V is non-zero, must not create "inconsistent" triples.
    
    # This is getting complex. Let me just try to find a low-rank decomposition
    # by solving a system of equations.
    
    # For a given rank r, we have:
    # U: 6 x r, V: 18 x r, W: 12 x r
    # 36 equations: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Plus constraints that no "wrong" entries are non-zero (but if the 36 equations are satisfied
    # and the decomposition is exact, the wrong entries should automatically be 0 if the
    # structure is correct)
    
    # Actually, we need to ensure that for all (a,b,c) NOT of the form (i*m+j, j*p+k, k*n+i),
    # sum_c U[a,c]*V[b,c]*W[c,c] = 0
    
    # This is a system of polynomial equations. Let me try to solve it for small r.
    
    # Start with r=18 and see if we can find a solution.
    
    # Actually, let me try a constructive approach first.
    
    # Observation: For each (j,k) pair, we need to handle 2 output entries (i=0 and i=1).
    # The minimal rank for handling a single (j,k) pair is 2 (one term for each i).
    # But maybe we can do better by sharing across k values for the same j.
    
    # For fixed j, we have 6 k-values and 2 i-values.
    # Can we use fewer than 12 terms for all (j,k) pairs with this j?
    
    # Idea: Use 6 terms for fixed j, where term k handles both i=0 and i=1?
    # Term k: U[0*m+j]=a, U[1*m+j]=b, V[j*p+k]=1, W[k*n+0]=c, W[k*n+1]=d
    # Contribution to T(0*m+j, j*p+k, k*n+0) = a*1*c = a*c
    # Contribution to T(1*m+j, j*p+k, k*n+1) = b*1*d = b*d
    # We need a*c = 1 and b*d = 1, so a=1/c, b=1/d
    # But we also need to ensure no cross terms:
    # T(0*m+j, j*p+k, k*n+1) = a*1*d = a*d = d/c (should be 0)
    # T(1*m+j, j*p+k, k*n+0) = b*1*c = b*c = c/d (should be 0)
    # So we need d/c = 0 and c/d = 0, which means c=0 or d=0, but then a*c=1 or b*d=1 is impossible.
    
    # So we can't combine i=0 and i=1 in a single term for fixed (j,k).
    
    # What about combining across k for fixed (j,i)?
    # For fixed (j,i), we have 6 k-values.
    # Term: U[i*m+j]=1, V[j*p+k1]=v1, V[j*p+k2]=v2, ..., W[k1*n+i]=w1, W[k2*n+i]=w2, ...
    # Contribution to T(i*m+j, j*p+k, k*n+i) = 1 * v_k * w_k
    # We need v_k * w_k = 1 for each k, so v_k = 1/w_k
    # Cross terms: T(i*m+j, j*p+k1, k2*n+i) = 1 * v_k1 * w_k2 = w_k2/w_k1 (should be 0)
    # So we need w_k2/w_k1 = 0 for k1 != k2, which means at most one w_k is non-zero.
    
    # So we can't combine across k either!
    
    # This suggests the rank is indeed 36 (one term per (i,j,k) triple).
    
    # But wait, maybe we can use linear combinations more cleverly.
    # Let me try a different approach: use the fact that the tensor has a "diagonal" structure.
    
    # Actually, I think the rank might be lower than 36. Let me try to find it numerically.
    
    # For now, let me return the schoolbook decomposition of rank 36.
    
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
# EVOLVE-BLOCK-END
