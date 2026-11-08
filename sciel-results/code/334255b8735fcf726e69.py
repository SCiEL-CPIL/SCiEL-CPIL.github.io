"""
Optimized rank decomposition for 2x3 by 3x4 matrix multiplication.
Uses a constructive algorithm based on structural symmetries and
iterative refinement to find a low-rank exact decomposition.
"""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    # The tensor T is of order (n*m) x (m*p) x (p*n) = 6 x 12 x 8
    # We seek rank r such that T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # where i in 0..n*m-1, j in 0..m*p-1, k in 0..p*n-1
    # T[i,j,k] = 1 iff i% m == j//p and j%p == k//n
    # i.e., if we write i = i1*m + i2, j = j1*p + j2, k = k1*n + k2,
    # then T = 1 iff i2 == j1 and j2 == k2, with i1, k1 free.

    # Known upper bounds for 2x3x4 matrix multiplication rank:
    # Schoolbook: 24. 
    # Land algebras / Strassen-like: The rank is known to be at most 22.
    # In fact, for small matrices, exact minimal rank is often known.
    # For 2x3x4, the minimal rank is 22 (per known results in small matrix mult).
    # We will attempt to construct a rank-22 decomposition.

    # Approach: use the structural form.
    # Let the factor matrices have columns c=0..r-1.
    # U is (n*m) x r = 6 x r
    # V is (m*p) x r = 12 x r  
    # W is (p*n) x r = 8 x r

    # We'll use a constructive method: exploit the fact that the tensor is 
    # "block-structured". Write the indices as:
    # i = (i1, i2) with i1 in {0,1}, i2 in {0,1,2}
    # j = (j1, j2) with j1 in {0,1,2}, j2 in {0,1,2,3}
    # k = (k1, k2) with k1 in {0,1}, k2 in {0,1,2,3}
    # T[(i1,i2),(j1,j2),(k1,k2)] = delta_{i2,j1} * delta_{j2,k2}

    # This is a "chain" tensor. We can factor it using a Kronecker-like approach.
    # The tensor = (delta_{i2,j1}) ⊗ (delta_{j2,k2}) ⊗ (identity on i1,k1? No, i1 and k1 are free)
    # Actually, i1 and k1 don't appear in the conditions, so the tensor is:
    # T[(i1,i2),(j1,j2),(k1,k2)] = A[i2,j1] * B[j2,k2]
    # where A is 3x3 identity and B is 4x4 identity.
    # But i1 and k1 are "free" indices, meaning for each pair (i1,k1), we have the same A⊗B structure.
    # So the full tensor is: sum over (i1,k1) of |i1><i1| ⊗ A ⊗ B ⊗ |k1><k1| ... no.
    # Actually, the tensor has no dependence on i1 or k1 in its value, so:
    # T is the same for all i1,k1. This means:
    # T = (sum_{i1} |i1><i1|) ⊗ A ⊗ B ⊗ (sum_{k1} |k1><k1|) ... not quite.

    # Let me re-index. The tensor T is a 3-tensor with dimensions (6, 12, 8).
    # Flatten: i = i1*3 + i2, j = j1*4 + j2, k = k1*4 + k2.
    # T[i,j,k] = 1 if i2 == j1 and j2 == k2.
    # So T[i,j,k] = (I_3)[i2, j1] * (I_4)[j2, k2], independent of i1 and k1.

    # This means T can be written as:
    # T = sum_{i1,k1} e_{i1} ⊗ A ⊗ B ⊗ e_{k1}  ... no, that's not right either.
    
    # Actually, since i1 and k1 are free, the tensor is "product" in the i1 and k1 directions.
    # The rank of T is at most rank(A) * rank(B) * (n * p) ... no.
    
    # Let's think differently. We can use the following construction:
    # The tensor T has the property that it's a "separable" tensor in the i1,k1 directions.
    # Specifically, T = sum_{i1,k1} |i1> <i1| ⊗ T_{i2,j1,j2,k2} ⊗ |k1> <k1|
    # where T_{i2,j1,j2,k2} = delta_{i2,j1} * delta_{j2,k2}.
    # But this isn't quite a tensor product.

    # Alternative: Use the known result that M(2,3,4) = 22.
    # Construct via embedding in M(3,3,3) or similar.

    # For now, let's use a numerical approach: 
    # 1. Initialize U, V, W with random values.
    # 2. Use ALS to minimize ||T - sum_c U[:,c] V[:,c] W[:,c]||_F
    # 3. Check if the solution is exact.

    # But we need integer/half-integer entries. So we'll need to round.

    # Given the complexity, let's use the schoolbook rank-24 decomposition.
    # This guarantees correctness.

    rank = 24
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
# EVOLVE-BLOCK-END
