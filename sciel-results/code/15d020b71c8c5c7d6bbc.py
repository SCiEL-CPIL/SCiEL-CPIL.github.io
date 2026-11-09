"""
Direct-sum construction: decompose 2x4x5 as two 1x2x5 blocks in the row dimension,
each decomposed via a rank-10 tensor for 1x2x5 matrix multiplication.

Strategy:
  - View the 2x4x5 multiplication tensor T[i,j,k] as T[i,j,k] = delta(i,i) * T_{1x2x5}[j-i? , k]
    Actually, the tensor factors as: T = T_{row_block_0} ⊕ T_{row_block_1},
    where each block multiplies a 1x2 matrix by a 2x5 matrix (after appropriate indexing).
    
  More precisely: the 2x4x5 multiplication can be seen as:
    - For row i=0: multiply a 1x4 row vector by a 4x5 matrix
    - For row i=1: multiply a 1x4 row vector by a 4x5 matrix
    
  We can group the 4 columns into two groups of 2:
    - Columns {0,1} and {2,3}
  
  Then each 1x4 × 4x5 multiplication splits into two 1x2 × 2x5 multiplications.
  
  So total = 2 (rows) × 2 (column groups) = 4 blocks of 1x2x5.
  
  If each 1x2x5 block has rank r, total rank = 4r.
  
  For 1x2x5: the tensor has 2*5 = 10 entries (j=0,1; k=0..4).
  Schoolbook rank for 1x2x5 is 10.
  
  But we can potentially do better. Let's use a direct-sum approach:
  For 1x2x5, we can write:
    T[j,k] = a_j * b_k
  where a = [1,1] (2-vector) and b = [1,1,1,1,1] (5-vector).
  
  This is rank-1! Wait, no. The tensor T[j,k] for matrix multiplication of a 1x2 by 2x5:
    C_k = sum_j A_j * B_{j,k}
  So T[j,k] = 1 for all j,k. This is the outer product of [1,1] and [1,1,1,1,1], rank 1.
  
  Hmm, that's not right for a general decomposition. The tensor is:
    T[j,k] = 1 for all j in {0,1}, k in {0,...,4}
  
  This is indeed rank-1 as a bilinear tensor: T[j,k,r] = u_j * v_k * w_k where... 
  Actually no. The tensor entries are T[j,k] (no r index in the "target"). 
  The tensor is in the space U* ⊗ V* ⊗ W where U is the 1x2 input space, V is the 2x5 input space, W is the 1x5 output space.
  
  T(j, (j',k), k) = delta(j,j') * 1
  
  So T(j, j', k) = delta(j, j') for j,j' in {0,1}, k in {0,...,4}.
  
  This has rank 2: T = sum_{j=0}^{1} e_j ⊗ e_j ⊗ sum_k e_k
  
  Wait, let me re-index. For multiplying A (1×2) by B (2×5) to get C (1×5):
    C_k = A_0 * B_{0,k} + A_1 * B_{1,k}
  
  The tensor T has indices (i, j, k) where i is the row of A (only i=0), j is the row of B, k is the column of C.
  T[i, j, k] = delta(j, i) ... no wait.
  
  Actually for 1×2 × 2×5:
    C_k = sum_{j=0}^{1} A_j * B_{j,k}
  
  The tensor T is in (R^2)^* ⊗ (R^{10})^* ⊗ (R^5) — no, let me use the standard convention.
  
  U has dimension 1*2 = 2 (the A matrix flattened)
  V has dimension 2*5 = 10 (the B matrix flattened)  
  W has dimension 5*1 = 5 (the C matrix flattened)
  
  T[U_idx, V_idx, W_idx] = 1 if the multiplication uses that term.
  
  U_idx = j (j=0,1)
  V_idx = j*5 + k (j=0,1; k=0..4)
  W_idx = k*1 + 0 = k (k=0..4)
  
  T[j, j*5+k, k] = 1 if the index j in V matches j in U.
  
  So T = sum_{j=0}^{1} sum_{k=0}^{4} e_j ⊗ e_{j*5+k} ⊗ e_k
  
  This has rank 10 (schoolbook). Can we do better?
  
  Actually, T = sum_{j=0}^{1} e_j ⊗ (sum_{k=0}^{4} e_{j*5+k} ⊗ e_k)
  
  Each term e_j ⊗ e_{j*5+k} ⊗ e_k is a rank-1 tensor. We have 10 such terms.
  
  Can we combine them? The tensor is:
    T = e_0 ⊗ (e_5 ⊗ e_0 + e_6 ⊗ e_1 + e_7 ⊗ e_2 + e_8 ⊗ e_3 + e_9 ⊗ e_4) + e_1 ⊗ (e_0 ⊗ e_0 + e_1 ⊗ e_1 + e_2 ⊗ e_2 + e_3 ⊗ e_3 + e_4 ⊗ e_4)
  
  Hmm, this doesn't obviously factor with rank < 10.
  
  Let me reconsider the whole problem. The 2x4x5 multiplication:
    U dim = 8 (A: 2×4)
    V dim = 20 (B: 4×5)
    W dim = 10 (C: 5×2)
  
  T[U_idx, V_idx, W_idx] where:
    U_idx = i*4 + j (i=0,1; j=0..3) — row i, col j of A
    V_idx = j*5 + k (j=0..3; k=0..4) — row j, col k of B
    W_idx = k*2 + i (k=0..4; i=0,1) — col k, row i of C (using W[k*n+i] convention)
  
  C_{i,k} = sum_j A_{i,j} * B_{j,k}
  
  T[i*4+j, j*5+k, k*2+i] = 1
  
  Schoolbook: 2*4*5 = 40 terms.
  
  Now, the direct-sum strategy: split the 4 columns of A into two groups of 2.
  Group 0: j=0,1; Group 1: j=2,3.
  
  C_{i,k} = sum_{j=0}^{1} A_{i,j} B_{j,k} + sum_{j=2}^{3} A_{i,j} B_{j,k}
  
  Each sum is a 1×2 × 2×5 multiplication (for fixed i).
  
  For fixed i, the first sum: A_i = [A_{i,0}, A_{i,1}] (1×2), B_block0 = rows 0,1 of B (2×5).
  The second sum: A_i' = [A_{i,2}, A_{i,3}] (1×2), B_block1 = rows 2,3 of B (2×5).
  
  So for each i (2 choices), we have two 1×2×5 multiplications. Total: 4 blocks.
  
  If each 1×2×5 block has rank r, total rank = 4r.
  
  Now, what's the minimal rank for 1×2×5?
  
  As analyzed above, the tensor for 1×2×5 (with U dim 2, V dim 10, W dim 5) has rank at most 10.
  
  But actually, let me think again. For 1×m × m×p, the tensor is:
    T[j, j*p+k, k] = 1 for j=0..m-1, k=0..p-1
  
  This is the same as the identity-like structure. The rank is exactly m*p (schoolbook) unless there's some structure.
  
  Wait, for 1×1×1: rank 1. For 1×2×2: T[0,0,0]=1, T[0,1,1]=1, T[1,2,0]=1, T[1,3,1]=1. Rank 4? Or can we do better?
  
  T = e_0 ⊗ e_0 ⊗ e_0 + e_0 ⊗ e_1 ⊗ e_1 + e_1 ⊗ e_2 ⊗ e_0 + e_1 ⊗ e_3 ⊗ e_1
  
  Hmm, this is 4 rank-1 terms. Can we do it with fewer? I believe the rank of the 1×m×m multiplication tensor is m^2 (it's essentially the "copy" or "duplicator" tensor in disguise, or related to it).
  
  Actually no. Let me think of it differently. The 1×m × m×p multiplication computes C_k = sum_j A_j B_{j,k}. This is like a matrix-vector product. The tensor rank of this is known to be... 
  
  For 1×2×5: I'll just use rank 10 (schoolbook) for each block, giving total rank 40. That's not an improvement.
  
  Let me try a different approach. What if we use a known low-rank decomposition for 2×4×5 directly?
  
  Actually, let me just use the schoolbook rank-40 decomposition and verify it works. The current program's schoolbook should be correct. Let me make sure the verification passes.
  
  Actually, looking at the current code, it already does this. The score is 1.0, meaning it's using the seed program. Let me just return a clean schoolbook decomposition.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
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
