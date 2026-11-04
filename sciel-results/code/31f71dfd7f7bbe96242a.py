import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    school_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # For matrix multiplication M_{n,m} * M_{m,p}, the tensor rank is known.
    # For small dimensions, we can try to find a decomposition with rank less than n*m*p.
    # The rank of the matrix multiplication tensor is at least max(n*m, m*p, p*n).
    # For 2,3,8: n*m=6, m*p=24, p*n=16. So rank >= 24.
    # Schoolbook rank is 48.
    # Let's try to find a decomposition of rank 24 if possible.
    
    # Actually, the rank of the tensor for matrix multiplication is not simply max of the flattening ranks.
    # Let's use a known result or try a systematic approach.
    # For 2x3x8, we can try rank = 24 (which is m*p).
    
    # Let's try to construct a rank-24 decomposition.
    # If we set r = m*p = 24, we need:
    # U: 6 x 24, V: 24 x 24, W: 16 x 24
    # such that the tensor decomposes correctly.
    
    # A known decomposition for matrix multiplication with rank = m*p (when p >= n)
    # can be constructed as follows:
    # Index the rank dimension by (j, k) where j in range(m), k in range(p).
    # U_{(i,j'), (j,k)} = delta_{j,j'}
    # V_{(j',k'), (j,k)} = delta_{j',j} * delta_{k',k} ... wait, this doesn't work directly.
    
    # Let me think more carefully.
    # T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # If we use rank r = m*p, indexed by (j, k):
    # U_{(i,j'), (j,k)} should be such that when summed with V and W, we get the deltas.
    
    # Let's try:
    # U_{(i,j'), (j,k)} = delta_{j,j'}
    # V_{(j',k'), (j,k)} = delta_{j',j} * delta_{k',k}  -- but V has indices (j',k') and (j,k)
    # Actually V is (m*p) x r = (m*p) x (m*p). So V_{(j',k'), (j,k)}.
    # W_{(i',k'), (j,k)} should complete the product.
    
    # Sum over (j,k) of U_{(i,j'), (j,k)} * V_{(j',k'), (j,k)} * W_{(i',k'), (j,k)}
    # = Sum over (j,k) of delta_{j,j'} * V_{(j',k'), (j,k)} * W_{(i',k'), (j,k)}
    # = Sum over k of V_{(j',k'), (j',k)} * W_{(i',k'), (j',k)}
    
    # We want this to equal delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # If we set V_{(j',k'), (j,k)} = delta_{j',j} (independent of k) * something...
    # This is getting complicated. Let's just return the schoolbook to be safe.
    # The rank is 48.
    
    return school_result
# EVOLVE-BLOCK-END
