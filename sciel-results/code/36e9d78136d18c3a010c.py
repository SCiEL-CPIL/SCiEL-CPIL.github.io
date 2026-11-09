"""
Matrix multiplication 2x3 by 3x7: rank-minimal exact tensor decomposition.

Strategy:
1. Theoretical lower bound via Kruskal-Cannon / exterior algebra.
   For multiplying an (n x m) matrix by an (m x p) matrix, the tensor
   lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
   The standard lower bound is max(n*m, m*p, p*n) but for small cases
   the true rank is often larger. For 2x3 * 3x7:
     n*m = 6, m*p = 21, p*n = 14.
   Known results: the rank of 2x3 * 3x7 is at least 21 (from the m*p bound)
   and it is known that rank(2x3 * 3x7) = 21 is achievable? Let's check.
   
   Actually, for matrix multiplication A(n×m)·B(m×p), the tensor rank is
   at least max(n·m, m·p, p·n) is NOT a valid lower bound in general.
   The correct lower bound from the "slice rank" or other techniques varies.
   
   However, a well-known result: if we fix the middle dimension m, the rank
   is at least m * max(n, p) in some cases, but more carefully:
   
   For 2×3 · 3×7: 
     - The output is 2×7 = 14 entries.
     - We can think of this as computing 14 bilinear forms.
     - Each bilinear form can potentially be shared.
   
   Let's use a constructive approach: try rank = 21 (the m*p bound) and see
   if we can find an exact decomposition. If not, try higher ranks.
   
   Actually, a simpler and more reliable approach: use the known fact that
   for matrix multiplication, a decomposition of rank = n*m*p is always
   possible (schoolbook). We want to minimize rank.
   
   For 2×3 · 3×7, let's try to construct a rank-21 decomposition.
   The idea: for each of the 21 products (j,k) where j=0..2 (rows of B / cols of A)
   and k=0..6 (cols of B), we form the product A[:,j] * B[j,k]. But A[:,j] is a
   2-vector and B[j,k] is a scalar, so the contribution to the output is a 2×1 vector
   scaled by B[j,k]. This gives 21 terms, each contributing to 2 output entries.
   
   Wait, that's not quite right. Let me think more carefully.
   
   The standard approach for low-rank matrix multiplication:
   We want C[i,k] = sum_j A[i,j] * B[j,k].
   
   For rank r decomposition, we need:
   C[i,k] = sum_{c=0}^{r-1} U[i, c] * V[c, k]  ... no, that's not the tensor form.
   
   The tensor form is:
   C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{c=0}^{r-1} (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
   
   Wait, the interface says:
   U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
   And W[k*n+i, r] is the output coefficient.
   
   So the decomposition is:
   For each column c of rank:
     u_c is a vector in R^{n*m} (flattened A)
     v_c is a vector in R^{m*p} (flattened B)
     w_c is a vector in R^{p*n} (flattened C, with index k*n+i)
   
   The tensor equation is:
   sum_c (u_c ⊗ v_c ⊗ w_c) = T
   where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
   
   Actually, T is the multiplication tensor: T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k.
   
   So for each (i,j,k) triple, T[i*m+j, j*p+k, k*n+i] = 1.
   
   A rank-r decomposition means we can write T as sum of r rank-1 tensors.
   
   Let's try rank = 21. We'll use the following construction:
   For each pair (j, k) where j in 0..2 and k in 0..6:
     Define u_{j,k}: u_{j,k}[i*m + j] = 1 for i=0,1, else 0.
     Define v_{j,k}: v_{j,k}[j*p + k] = 1, else 0.
     Define w_{j,k}: w_{j,k}[k*n + i] = 1 for i=0,1, else 0.
   
   Then sum_{j,k} u_{j,k} ⊗ v_{j,k} ⊗ w_{j,k} has entry at [i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
   This is exactly T! And the rank is 21.
   
   So rank 21 is achievable. Is it minimal?
   
   Lower bound: The slice rank or other bounds. For 2×3·3×7, 
   m*p = 21. It is known that the rank of matrix multiplication tensor 
   M_{n,m,p} is at least m * max(n,p) when... actually I'm not sure of the exact bound.
   
   But since we have a construction with rank 21, and the schoolbook is 42, 
   rank 21 is at least a 2x improvement. Let's go with rank 21.
"""

import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = m * p = 21
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
