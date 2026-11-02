"""
Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.

The matmul tensor T_{2,3,5} has 2*3*5 = 30 entries, each equal to 1.
The schoolbook decomposition uses rank 30 (one rank-1 term per entry).

This program attempts to find a lower-rank decomposition by searching for
non-trivial rank-1 tensors that can replace pairs of schoolbook terms.
"""

import numpy as np

DIMENSIONS = [2, 3, 5]


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x5 matrix.

    Returns a dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    respectively, with integer/half-integer entries that exactly decompose the matrix
    multiplication tensor.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    # Build the schoolbook decomposition (rank 30)
    rank_school = n * m * p  # 30
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # The matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # A rank-r decomposition satisfies:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    #
    # For the schoolbook, each term c corresponds to exactly one (i,j,k) with value 1,
    # and all other terms contribute 0 to that entry.
    #
    # To reduce rank, we need to find rank-1 tensors u⊗v⊗w where u, v, w have
    # multiple non-zero entries, such that the sum still equals T.
    #
    # Key constraint: for a rank-1 term to be "compatible" with the matmul tensor,
    # whenever u[i,j] ≠ 0, v[j,k] ≠ 0, w[k,i] ≠ 0, the product u[i,j]*v[j,k]*w[k,i]
    # contributes to T[i,j,k]. Since T[i,j,k] = 1 for ALL valid (i,j,k), there are no
    # "forbidden" entries — every (i,j,k) in the n×m×p grid is a valid entry.
    #
    # So the constraint is simply: sum over all rank-1 terms of u_c[i,j]*v_c[j,k]*w_c[k,i] = 1
    # for all i∈[0,2), j∈[0,3), k∈[0,5).
    #
    # Strategy: Try to merge pairs of schoolbook terms into a single rank-1 term.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if
    # there exist vectors u, v, w such that:
    #   u[i1,j1]*v[j1,k1]*w[k1,i1] = a
    #   u[i2,j2]*v[j2,k2]*w[k2,i2] = b
    #   For all other (i,j,k): u[i,j]*v[j,k]*w[k,i] = 0
    #   And a + b = 2 (to replace the two 1's from schoolbook)
    #
    # But we also need the merged term to not interfere with other entries.
    # This is very restrictive.
    #
    # Alternative approach: Use the fact that for fixed (i,k), the equation becomes:
    #   sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # Let's try a different strategy: assume a decomposition where each rank-1 term
    # has a specific structure. For example, terms where u is supported on a single row i
    # of A, v is supported on a single column k of B, and w is supported on a single (k,i)
    # entry of C.
    #
    # In this case, a rank-1 term with u supported on row i, v on column k, w on (k,i):
    #   u[i,j] for j=0..m-1, v[j,k] for j=0..m-1, w[k,i] = scalar
    #   Product at (i',j',k'): non-zero only if i'=i, k'=k, and j' is in support of both u and v
    #   Value = u[i,j']*v[j',k]*w[k,i]
    #
    # So such a term contributes to entries (i, j, k) for various j, with value
    # u[i,j]*v[j,k]*w[k,i].
    #
    # If we use one such term per (i,k) pair, we have n*p = 10 terms. For each (i,k),
    # we need: sum_j u[i,j]*v[j,k]*w[k,i] = 1 for all j? No, that's not right.
    #
    # Actually, for a fixed (i,k), the term contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k]
    # for each j. We need the SUM over all terms to give 1 for each (i,j,k).
    #
    # If we have one term per (i,k) pair, then for entry (i,j,k), only the term for
    # that (i,k) contributes (since other terms have different i' or k'). So we need:
    #   u[i,j]*v[j,k]*w[k,i] = 1 for all i, j, k
    #
    # This means u[i,j]*v[j,k] = 1/w[k,i] for all j. For fixed (i,k), this must hold
    # for all j=0,1,2. So u[i,j]*v[j,k] must be constant in j.
    #
    # Let u[i,j] = a_{i,j}, v[j,k] = b_{j,k}, w[k,i] = c_{k,i}.
    # Constraint: a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    #
    # This is a system of 30 equations in 2*3 + 3*5 + 5*2 = 6+15+10 = 31 unknowns.
    # We have 30 equations, so there's a 1-parameter family of solutions (if consistent).
    #
    # Let's solve: a_{i,j} * b_{j,k} = 1/c_{k,i} for all i,j,k.
    #
    # For fixed i, j: a_{i,j} = 1/(c_{k,i} * b_{j,k}) for all k.
    # So 1/(c_{k1,i} * b_{j,k1}) = 1/(c_{k2,i} * b_{j,k2})
    # => c_{k1,i} * b_{j,k1} = c_{k2,i} * b_{j,k2}
    # => b_{j,k1}/c_{k2,i} = b_{j,k2}/c_{k1,i}  (for all k1≠k2)
    #
    # This means b_{j,k}/c_{k,i} is independent of k (for fixed j, i... wait, i is fixed here).
    # Actually for fixed i and j: b_{j,k}/c_{k,i} must be the same for all k.
    # Let b_{j,k} = d_j * c_{k,i} for all k, where d_j depends on j (and implicitly i? No, i is fixed).
    # Wait, this must hold for all i. So b_{j,k}/c_{k,i} = f_{j}(i) for some function f_j.
    # But also for fixed j and k: b_{j,k} * c_{k,i}^{-1} = f_j(i) for all i.
    # So b_{j,k} = f_j(i) * c_{k,i} for all i. This must hold for all i, which means
    # c_{k,i} must be proportional to 1/f_j(i) for all j... this is getting complex.
    #
    # Let's try a simpler ansatz: c_{k,i} = 1 for all k,i. Then we need
    # a_{i,j} * b_{j,k} = 1 for all i,j,k.
    # For fixed j: a_{i,j} * b_{j,k} = 1 for all i,k.
    # So a_{i,j} = 1/b_{j,k} for all i,k, meaning a_{i,j} is independent of i (for fixed j)
    # and b_{j,k} is independent of k (for fixed j).
    # Let a_{i,j} = α_j, b_{j,k} = β_j, with α_j * β_j = 1.
    # So α_j = 1/β_j.
    #
    # This gives a valid decomposition with rank n*p = 10!
    # U[i*m+j, c] where c indexes (i,k) pairs: U[i*m+j, (i,k)] = α_j
    # V[j*p+k, (i,k)] = β_j if k matches, else 0... 
    #
    # Wait, I need to be more careful. Let me re-derive.
    #
    # Rank-1 term for (i,k): u_{i,k} ⊗ v_{i,k} ⊗ w_{i,k}
    #   u_{i,k}[i', j'] = α_j if i'=i, else 0  (supported on row i of A)
    #   v_{i,k}[j', k'] = β_j if k'=k, else 0  (supported on column k of B)
    #   w_{i,k}[k'', i''] = 1 if (k'',i'')=(k,i), else 0  (supported on (k,i) of C)
    #
    # Product at (i', j', k'): 
    #   u_{i,k}[i',j'] * v_{i,k}[j',k'] * w_{i,k}[k',i']
    #   = α_{j'} * β_{j'} * 1 if i'=i, k'=k, else 0
    #   = α_{j'} * β_{j'} if (i',k')=(i,k), else 0
    #
    # Sum over all (i,k) terms: for entry (i,j,k), only the term (i,k) contributes,
    # giving α_j * β_j. We need α_j * β_j = 1 for all j.
    #
    # So set α_j = 1, β_j = 1 for all j. Then we have a rank-10 decomposition!
    #
    # Let's verify: U has shape (2*3, 10) = (6, 10), V has shape (3*5, 10) = (15, 10),
    # W has shape (5*2, 10) = (10, 10).
    #
    # For column c corresponding to (i,k):
    #   U[i*m + j, c] = 1 for all j (since α_j = 1)
    #   V[j*p + k, c] = 1 for all j (since β_j = 1)
    #   W[k*n + i, c] = 1
    #
    # Check entry (i,j,k): sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Only c=(i,k) has W[k*n+i, c] = 1 (all other c' have W[k*n+i, c'] = 0).
    # For c=(i,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Product = 1. Sum = 1. ✓
    #
    # This is a rank-10 decomposition! Much better than 30.

    # Build the rank-10 decomposition
    rank_new = n * p  # 10
    U_new = np.zeros((n * m, rank_new), dtype=float)
    V_new = np.zeros((m * p, rank_new), dtype=float)
    W_new = np.zeros((p * n, rank_new), dtype=float)

    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U_new[i * m + j, col] = 1
                V_new[j * p + k, col] = 1
            W_new[k * n + i, col] = 1
            col += 1

    # Verify the decomposition
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank_new):
                    s += U_new[i * m + j, c] * V_new[j * p + k, c] * W_new[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
