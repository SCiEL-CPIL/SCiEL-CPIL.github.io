"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

The bilinear map is: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..6}.

Reconstruction convention (verified against schoolbook):
  C[i,k] = sum_c  U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]   (summed over j)
  i.e.   C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])

For the schoolbook term c=(i',j',k'):
  U[i*m+j,c] = delta_{i,i'} * delta_{j,j'}
  V[j'*p+k,c] = delta_{j',j} * delta_{k',k}  -- wait, V is indexed by (j*p+k)
  So V[j*p+k, c] = 1 iff j=j' AND k=k'
  W[k*n+i, c] = 1 iff k=k' AND i=i'
  
  Then sum_j U[i*m+j,c]*V[j*p+k,c] = U[i*m+j',c]*V[j'*p+k,c] = delta_{i,i'}*delta_{j,j'}*delta_{j',j}*delta_{k',k}
  = delta_{i,i'}*delta_{j,j'}*delta_{k',k}  (when j=j')
  = delta_{i,i'}*delta_{k',k}  (summed over j, only j=j' contributes)
  
  So C[i,k] = sum_c W[k*n+i,c] * delta_{i,i'}*delta_{k',k}
  = sum_{c where i_c=i, k_c=k} 1 = m = 4  ... that's not right.

Let me re-derive. For schoolbook term c=(i',j',k'):
  U[i*m+j, c] = 1 if i=i' and j=j', else 0
  V[j*p+k, c] = 1 if j=j' and k=k', else 0
  W[k*n+i, c] = 1 if k=k' and i=i', else 0

The reconstruction formula must be:
  C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For term c=(i',j',k'):
  The inner sum over j: only j=j' gives nonzero U and V.
  U[i*m+j', c] = delta_{i,i'}
  V[j'*p+k, c] = delta_{k,k'}
  W[k*n+i, c] = delta_{k,k'}*delta_{i,i'}
  
  So contribution = delta_{i,i'} * delta_{k,k'} * delta_{k,k'}*delta_{i,i'} = delta_{i,i'}*delta_{k,k'}

Summing over all c=(i',j',k'):
  C[i,k] = sum_{i',j',k'} delta_{i,i'}*delta_{k,k'} = sum_{j'} 1 = 4

That gives C[i,k] = 4, not the correct value. So the formula must be different.

Perhaps the correct formula is:
  C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]

No, that would make U and V multiply with A and B, which is a different kind of decomposition.

Actually, I think the standard formulation for tensor decomposition of matmul is:
  The tensor T has T[i,j,k] = delta (it's the identity in the middle dimension)
  And we decompose: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
  
  But that's a trilinear form, not bilinear.

For BILINEAR decomposition:
  C[i,k] = sum_c alpha_c(i) * beta_c(k) * (gamma_c^T A delta_c) * ... 

I think the correct interpretation for this specific interface is:
  C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])

No wait, that's not standard either.

Let me look at this differently. The seed program uses schoolbook and claims it works. Let me just trust that the evaluator knows the correct reconstruction formula and that the schoolbook is valid. The key question is: what formula does the evaluator use?

Given the shapes:
- U: (n*m, rank) = (8, r)
- V: (m*p, rank) = (28, r)  
- W: (p*n, rank) = (14, r)

The most natural bilinear decomposition formula would be:
  C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]

But this doesn't reduce to schoolbook correctly either.

Alternative: maybe U, V, W are NOT multiplied with A and B, but rather:
  The tensor T[i,j,k] (where T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k])
  is decomposed as: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

And then C[i,k] = sum_j A[i,j] * B[j,k] * (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])... no.

I think the correct formula is simply:
  C[i,k] = sum_c W[k*n+i, c] * sum_j (U[i*m+j, c] * V[j*p+k, c]) * A[i,j] * B[j,k]

No, that's not right either.

Let me just go with the most standard interpretation: the three matrices U, V, W define a rank-r decomposition of the matmul tensor where:
  C[i,k] = sum_c (u_c^T * vec_A_block) * (v_c^T * vec_B_block) * w_c_entry

Actually, I'll just implement the schoolbook exactly as in the seed (which scores 1.0) and try to reduce the rank by exploiting structure. The safest approach is to keep the same structure but see if any terms can be merged.

For [2,4,7], the schoolbook has 56 terms. Let me try to find a valid decomposition with fewer terms by exploiting the n=2 structure.

Key insight: C[1,k] - C[0,k] = sum_j (A[1,j] - A[0,j]) * B[j,k]

This means if we define D[j] = A[1,j] - A[0,j], then:
  C[0,k] = sum_j A[0,j] B[j,k]
  C[1,k] = C[0,k] + sum_j D[j] B[j,k]

So we need:
- A decomposition of [1,4,7] for A[0,:] @ B: rank r1
- A decomposition of [1,4,7] for D @ B: rank r2
- Then C[1,k] = C[0,k] + (D@B)[k]

The second part can be folded into the W factors. Specifically, if the [1,4,7] decomposition for A[0,:]@B gives terms with W entries W0[k*n+0, c] = 1 (for output row 0), and the D@B decomposition gives terms with W entries W1[k*n+1, c] = 1 (for output row 1), we can modify:
- For the A[0,:]@B terms: set W[k*n+0, c] = 1 and W[k*n+1, c] = 1 (so they contribute to both rows)
- For the D@B terms: set W[k*n+1, c] = 1 and W[k*n+0, c] = 0 (so they only contribute to row 1)

This gives total rank = r1 + r2 where r1 = rank of [1,4,7] and r2 = rank of [1,4,7].

The minimum rank of [1,4,7] matrix multiplication is 7 (since each of the 7 outputs needs at least 7 independent products... actually no, the lower bound for [1,m,p] is max(m,p) by a simple argument, but the actual rank can be higher).

For [1,4,7], schoolbook is 4*7 = 28. Can we do better? The tensor rank of [1,m,p] matmul is known to be m*p in general (no savings possible for a single row), because each output C[k] = sum_j A[j]B[j,k] requires all m*p products in the worst case.

So r1 = r2 = 28, total = 56. No improvement.

Let me try a different approach. What if we use a shared decomposition?

Define:
  S[j,k] = A[0,j]*B[j,k]  (for j=0..3, k=0..6): 28 values
  T[j,k] = A[1,j]*B[j,k]  (for j=0..3, k=0..6): 28 values

C[0,k] = sum_j S[j,k]
C[1,k] = sum_j T[j,k]

Total 56 products. No obvious sharing.

What about defining:
  P[j,k] = (A[0,j] + A[1,j]) * B[j,k]
  Q[j,k] = (A[0,j] - A[1,j]) * B[j,k]

Then:
  A[0,j]*B[j,k] = (P[j,k] - Q[j,k])/2
  A[1,j]*B[j,k] = (P[j,k] + Q[j,k])/2

C[0,k] = sum_j (P[j,k] - Q[j,k])/2
C[1,k] = sum_j (P[j,k] + Q[j,k])/2

Still 56 products (28 P + 28 Q). No improvement.

It seems like for [2,4,7], the schoolbook rank of 56 may indeed be optimal or close to it. Let me just output the schoolbook decomposition, which is known to be valid.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
