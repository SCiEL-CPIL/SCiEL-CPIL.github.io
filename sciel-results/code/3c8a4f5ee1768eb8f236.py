import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the matrix-multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j'*p + k, k'*n + i'] = 1 if i==i', j==j', k==k' else 0
    # We want to decompose T as sum_t U[:,t] V[:,t] W[:,t] (outer product of 3 vectors)
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    
    # The tensor is a "block diagonal" structure: for each j, we have an identity-like
    # coupling between (i, j) on U, (j, k) on V, and (k, i) on W.
    # Specifically, T = sum_j (I_n on U-side) ⊗ (I_m on V-side) ⊗ (I_p on W-side)
    # where the j-index selects which block.
    
    # Key insight: The tensor can be written as:
    # T_{(ij), (j'k), (i'k')} = δ_{ii'} δ_{jj'} δ_{kk'}
    
    # This is the tensor for multiplying (n×m) × (m×p) → (n×p).
    # The rank of this tensor is known to be n*m*p in general, but there may be
    # shortcuts.
    
    # Let's think about it differently. Consider the tensor as a 3-way array.
    # The "schoolbook" decomposition has rank n*m*p = 2*3*8 = 48.
    
    # Can we do better? The lower bound from flattening is max(n, m, p) = 8.
    # But that's a very weak bound.
    
    # Actually, for matrix multiplication tensor M_{n,m,p}, the rank is exactly
    # n*m*p for small cases, but there are known results that it can be lower
    # for some dimensions.
    
    # However, the tensor here is specifically the "identity" tensor:
    # T_{(ij), (j'k), (i'k')} = δ_{ii'} δ_{jj'} δ_{kk'}
    
    # This can be written as:
    # sum_{j=1}^{m} [ (e_j in U-space) ⊗ (e_j in V-space) ⊗ (identity coupling in W) ]
    
    # More precisely, for each fixed j, the sub-tensor is:
    # T_{(ij), (jk), (i'k')} with j fixed = δ_{ii'} δ_{kk'}
    # This is the tensor for multiplying (n×1) × (1×p) → (n×p), which has rank n*p.
    
    # So the total rank is at most m * n * p = 48 (schoolbook).
    
    # But wait - can we share terms across different j values?
    # If we use a term U_col, V_col, W_col, it contributes to all (i,j),(j',k),(i',k')
    # where U has non-zero at (i,j), V has non-zero at (j',k), W has non-zero at (i',k').
    # For this to match δ_{ii'}δ_{jj'}δ_{kk'}, we need:
    # - U has support only on indices with a specific j
    # - V has support only on indices with the same j
    # - W has support on (i',k') pairs that match the U and V supports
    
    # Actually, let me reconsider. The tensor is:
    # T_{(ij), (j'k), (i'k')} = δ_{ii'} δ_{jj'} δ_{kk'}
    
    # A rank-1 term is U_{(ij),t} * V_{(j'k),t} * W_{(i'k'),t}
    
    # For the sum over t to equal δ_{ii'} δ_{jj'} δ_{kk'}, we need that for each
    # (i,j,k,i',j',k'), the sum is 1 only when i=i', j=j', k=k'.
    
    # This is equivalent to: the matrix T flattened in mode 1 vs (2,3) has rank
    # at most r, where r is the tensor rank.
    
    # Let me try a constructive approach. Consider:
    # For each pair (j, k), define a term that "handles" the coupling.
    
    # Actually, let me think about it as: for each j, we need to decompose the
    # n×p identity-like tensor (which is just the standard basis coupling).
    
    # The n×p "multiplication" tensor for (n×1)×(1×p)→(n×p) is:
    # T_{(i,j), (j,k), (i',k')} = δ_{ii'} δ_{kk'} for fixed j
    # This has rank n*p (each (i,k) pair needs its own term).
    
    # So total rank = m * n * p = 48.
    
    # Hmm, but maybe we can do better by using half-integers or clever overlaps?
    
    # Actually, I recall that the rank of the matrix multiplication tensor M_{n,m,p}
    # is n*m*p when the tensor is the "full" one. But here our tensor is specifically
    # the one where T_{(ij),(j'k),(i'k')} = δ_{ii'}δ_{jj'}δ_{kk'}, which IS the
    # standard matrix multiplication tensor.
    
    # Wait, no. The standard matrix multiplication tensor is:
    # T_{(i,k), (j,k'), (i',j')} = δ_{ij} δ_{kk'} for C_{ij} = sum_k A_{ik} B_{kj}
    # That's different from what we have here.
    
    # Our tensor: U indices are (i,j) where i∈[1,n], j∈[1,m]
    # V indices are (j,k) where j∈[1,m], k∈[1,p]
    # W indices are (k,i) where k∈[1,p], i∈[1,n]
    # T_{(ij),(j'k),(i'k')} = δ_{ii'} δ_{jj'} δ_{kk'}
    
    # This means: C_{i'k'} = sum_{ij, j'k} U_{(ij),t} V_{(j'k),t} W_{(i'k'),t}
    # For this to compute C_{ik} = sum_j A_{ij} B_{jk}, we need:
    # C_{i'k'} should equal sum_j A_{ij} B_{jk} when i'=i, k'=k, and 0 otherwise.
    
    # So the W tensor must have the property that W_{(i'k'),t} selects the right
    # (i',k') output position.
    
    # Let me just construct the schoolbook solution and verify it.
    
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    # Check that sum_t U_{(ij),t} V_{(j'k),t} W_{(i'k'),t} = δ_{ii'} δ_{jj'} δ_{kk'}
    def verify(U, V, W, n, m, p):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                val = sum(U[i*m+j][t] * V[j2*p+k][t] * W[k2*n+i2][t] for t in range(r))
                                expected = (1 if i==i2 and j==j2 and k==k2 else 0)
                                if abs(val - expected) > 1e-10:
                                    return False, (i,j,k,i2,j2,k2,val,expected)
        return True, None
    
    valid, info = verify(U, V, W, n, m, p)
    if not valid:
        print(f"Verification failed: {info}")
    
    return result

# EVOLVE-BLOCK-END
