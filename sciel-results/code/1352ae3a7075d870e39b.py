"""Matrix multiplication tensor decomposition 3x5x7 with block-diagonal row-wise construction."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Block-diagonal decomposition:
    # Treat A as a stack of n row vectors, each of length m.
    # For row i of A, we compute C[i, :] = A[i, :] * B
    # This is a 1 x m times m x p matrix multiplication.
    # The rank of a 1 x m x p tensor (row vector times matrix) is m (for m <= p).
    # We can decompose A[i, :] * B = sum_{j=0}^{m-1} A[i,j] * (e_j^T * B)
    # where e_j is the j-th standard basis vector of length m.
    # e_j^T * B is the j-th row of B, which is a 1 x p vector.
    # So C[i, k] = sum_{j=0}^{m-1} A[i,j] * B[j, k]
    #
    # Tensor decomposition:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} u_{i,j} ⊗ v_{i,j} ⊗ w_{i,j}
    # where:
    #   u_{i,j}[a] = 1 if a = i*m + j, else 0
    #   v_{i,j}[b] = 1 if b = j*p + k for some k... 
    #
    # Wait, let me think about this more carefully.
    #
    # The matrix multiplication tensor T has indices (a, b, c) where:
    #   a = i*m + j (row i, col j of A)
    #   b = j'*p + k (row j', col k of B)  
    #   c = k'*n + i' (row k', col i' of C)
    #
    # T[a, b, c] = 1 if and only if:
    #   j = j' (the column of A must match the row of B)
    #   k = k' (the column of B must match the column of C)
    #   i = i' (the row of A must match the row of C)
    #
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i, j, k), and 0 otherwise.
    #
    # The trivial decomposition has rank n*m*p with one term per (i,j,k).
    #
    # Can we do better? The key insight is that for fixed i, the terms with the same i
    # can potentially share structure. But actually, the tensor is a direct sum over i
    # of n tensors, each of size (m, m*p, p*n) restricted to the i-th block.
    #
    # For fixed i, the sub-tensor T_i has:
    #   T_i[j, j*p+k, k*n+i] = 1 for all j, k
    #   This is the tensor for multiplying a 1 x m row vector by an m x p matrix.
    #
    # The rank of the 1 x m x p matrix multiplication tensor is m (when m <= p).
    # Proof: C[k] = sum_j A[j] * B[j, k]
    # We can write this as: for each j, define a term with:
    #   u_j = e_j (the j-th basis vector of R^m)
    #   v_j = the j-th row of B... but B is a variable, not a fixed matrix.
    #
    # Actually, the tensor rank of the 1 x m x p matmul tensor:
    # T_{j, j*p+k, k} = 1 (using 0-indexing for the 1 x m x p case)
    # This tensor has rank m because:
    #   T = sum_{j=0}^{m-1} u_j ⊗ v_j ⊗ w_j
    #   where u_j = e_j in R^m
    #   v_j has 1 at position j*p + k for all k... no, that's not right either.
    #
    # Let me reconsider. For the 1 x m x p case:
    #   Input: A is 1 x m, B is m x p, Output C is 1 x p
    #   C[k] = sum_j A[j] * B[j, k]
    #
    # Tensor indices: (a, b, c) where a in [0,m), b in [0, m*p), c in [0, p)
    #   T[a, b, c] = 1 if b = a*p + c, else 0
    #   i.e., T[a, a*p+c, c] = 1
    #
    # Rank of this tensor: We need sum_t u_t[a] * v_t[b] * w_t[c] = T[a, b, c]
    # For b = a*p + c: sum_t u_t[a] * v_t[a*p+c] * w_t[c] = 1
    # For b != a*p + c: sum_t u_t[a] * v_t[b] * w_t[c] = 0
    #
    # A rank-m decomposition:
    #   For t = 0, ..., m-1:
    #     u_t = e_t (basis vector in R^m)
    #     v_t has v_t[t*p + c] = 1 for all c in [0,p), and 0 otherwise
    #     w_t = ones(p) (all ones in R^p)
    #
    # Check: sum_t u_t[a] * v_t[b] * w_t[c] = u_a[a] * v_a[b] * 1 (only t=a survives)
    #   = 1 if b = a*p + c, else 0. Correct!
    #
    # So the rank of the 1 x m x p matmul tensor is m.
    #
    # Now for the full n x m x p case:
    #   The tensor is a direct sum over i of the 1 x m x p tensors (with appropriate indexing).
    #   Total rank = n * m = 3 * 5 = 15.
    #
    # Construction:
    #   For i in [0, n), j in [0, m):
    #     Term index t = i*m + j
    #     u_t[a] = 1 if a = i*m + j, else 0
    #     v_t[b] = 1 if b = j*p + k for some k in [0, p), else 0
    #     w_t[c] = 1 if c = k*n + i for some k in [0, p), else 0
    #
    # Wait, let me verify this works.
    #
    # sum_t u_t[a] * v_t[b] * w_t[c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    #
    # u_t[a] = 1 only if t corresponds to (i, j), i.e., t = i*m + j
    # So only t = i*m+j survives:
    #   u_{i*m+j}[i*m+j] = 1
    #   v_{i*m+j}[b] = 1 if b = j*p + k' for some k'
    #   w_{i*m+j}[c] = 1 if c = k''*n + i for some k''
    #
    # So the sum is:
    #   1 * (1 if b = j*p + k') * (1 if c = k''*n + i)
    #   = 1 if j = j' and i = i'', else 0
    #
    # But we need this to equal T[a, b, c] which is 1 if j = j' AND k' = k'' AND i = i''.
    #
    # The issue is that k' and k'' are not constrained to be equal in my construction.
    # v_t[b] = 1 for all k' when b = j*p + k', and w_t[c] = 1 for all k'' when c = k''*n + i.
    # So the product is 1 whenever j = j' and i = i'', regardless of whether k' = k''.
    #
    # This means my construction gives:
    #   sum_t u_t[a] * v_t[b] * w_t[c] = 1 if j = j' and i = i'', else 0
    #
    # But T[a, b, c] = 1 only if j = j' AND k' = k'' AND i = i''.
    #
    # So my construction is WRONG. It overcounts.
    #
    # I need w_t to depend on k as well. Let me reconsider.
    #
    # For the 1 x m x p case, the correct decomposition was:
    #   u_t = e_t, v_t has 1 at positions t*p + c for all c, w_t = ones(p)
    #   This works because in the 1 x m x p case, there's only one row i, so
    #   the output index c directly corresponds to k.
    #
    # In the full n x m x p case, the output index c = k*n + i includes both k and i.
    # For fixed i, the output positions are c = k*n + i for k = 0, ..., p-1.
    #
    # So for the i-th block, the 1 x m x p tensor has:
    #   Input A: m entries (indices 0 to m-1 within the block)
    #   Input B: m*p entries, but only the j-th row matters for the j-th input
    #   Output C: p entries (positions k*n + i for k = 0, ..., p-1)
    #
    # The tensor for the i-th block: T_i[j, j*p+k, k*n+i] = 1
    #
    # This is exactly the 1 x m x p tensor with output indices relabeled as k*n + i instead of k.
    # The rank is still m.
    #
    # Decomposition for the i-th block:
    #   For j = 0, ..., m-1:
    #     u_{i,j}[a] = 1 if a = i*m + j, else 0
    #     v_{i,j}[b] = 1 if b = j*p + k for some k, else 0
    #     w_{i,j}[c] = 1 if c = k*n + i for some k, else 0
    #
    # But as I showed, this gives 1 whenever j = j' and i = i'', not requiring k = k''.
    #
    # The problem is that v_{i,j} has 1s at ALL positions j*p + k for all k, and
    # w_{i,j} has 1s at ALL positions k'*n + i for all k'. The product doesn't enforce k = k'.
    #
    # I need a different approach. Let me think again about the 1 x m x p case.
    #
    # In the 1 x m x p case, T[a, b, c] = 1 if b = a*p + c.
    # The decomposition:
    #   u_t = e_t, v_t has 1 at t*p + c for all c, w_t = ones(p)
    #   sum_t u_t[a] v_t[b] w_t[c] = u_a[a] v_a[b] * 1 = v_a[b]
    #   v_a[b] = 1 if b = a*p + c, else 0. Correct.
    #
    # The key is that in this case, the output index c is directly the "k" index,
    # and v_t depends on both the input index t and the output index c.
    #
    # In the full case, for fixed i, the output index is c = k*n + i.
    # The "k" is embedded in c. So I need:
    #   v_{i,j}[b] should be 1 only at b = j*p + k where k is determined by c.
    #   But v doesn't depend on c, so this is impossible with a single v vector.
    #
    # This means the block-diagonal approach with rank n*m = 15 does NOT work directly.
    #
    # Let me reconsider. The correct rank for the 1 x m x p tensor is m, but the
    # decomposition requires v_t to depend on the output index in a specific way.
    #
    # Actually, I think I made an error. Let me re-examine.
    #
    # In the 1 x m x p case:
    #   T[a, b, c] = 1 iff b = a*p + c
    #   Decomposition: u_t = e_t, v_t[b] = 1 iff b = t*p + c (for any c), w_t[c] = 1
    #
    # But v_t[b] = 1 iff b = t*p + c for SOME c. Since b ranges over [0, m*p),
    # and c ranges over [0, p), the condition b = t*p + c means b is in the range
    # [t*p, t*p + p - 1]. So v_t has 1s in the t-th block of p consecutive entries.
    #
    # Now in the full n x m x p case, for fixed i:
    #   T_i[j, b, c] = 1 iff b = j*p + k AND c = k*n + i
    #   i.e., b = j*p + k AND c = k*n + i, which means k = b - j*p = (c - i)/n
    #   So we need b - j*p = (c - i)/n, i.e., n*(b - j*p) = c - i
    #
    # The decomposition for the i-th block should be:
    #   For j = 0, ..., m-1:
    #     u_{i,j}[a] = 1 iff a = i*m + j
    #     v_{i,j}[b] = 1 iff b is in [j*p, j*p + p-1] (the j-th block of B)
    #     w_{i,j}[c] = 1 iff c is in [i, i+p*n] with step n, i.e., c = k*n + i for some k
    #
    # But this gives the same problem: the product v * w is 1 whenever b is in the
    # j-th block AND c is in the i-th subsequence, but doesn't enforce the correspondence
    # between the specific k values.
    #
    # I think the issue is that the 1 x m x p tensor rank is m, but when embedded in the
    # larger tensor, the cross-terms between different i values might cause issues.
    #
    # Actually, let me just try the trivial rank 105 solution, which is guaranteed correct.
    # The task says to find a rank-minimal decomposition, but if I can't find a better one,
    # I should return a valid one.
    
    trivial_rank = n * m * p  # 105
    
    U_list = [[0] * trivial_rank for _ in range(n * m)]
    V_list = [[0] * trivial_rank for _ in range(m * p)]
    W_list = [[0] * trivial_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_list[a][col] = 1
                V_list[b][col] = 1
                W_list[c][col] = 1
                col += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
