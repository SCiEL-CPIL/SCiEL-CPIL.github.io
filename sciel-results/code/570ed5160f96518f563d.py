"""Tensor decomposition for 3x4 @ 4x8 matrix multiplication using block Strassen."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-48 decomposition of the 3x4 @ 4x8 multiplication tensor.
    
    Strategy: Use the fact that the 3x2 @ 2x4 multiplication tensor is the
    all-ones tensor of shape (3,2,4) which has rank min(3*2, 2*4, 4*3) = 6.
    
    For 3x4 @ 4x8:
      C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Split j into two groups of 2: j in {0,1} and j in {2,3}.
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    The first term is a 3x2 @ 2x8 multiplication.
    The second term is a 3x2 @ 2x8 multiplication.
    
    For 3x2 @ 2x8: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is the all-ones tensor of shape (3,2,8) with rank min(3*2, 2*8, 8*3) = min(6,16,24) = 6.
    
    So each 3x2 @ 2x8 uses 6 products, and total rank = 2 * 6 = 12? 
    Wait, that seems too good. Let me verify.
    
    Actually, the all-ones tensor T[i,j,k] = 1 for i in 0..n-1, j in 0..m-1, k in 0..p-1
    has tensor rank = min(n*m, m*p, p*n) when n,m,p are such that one of these is smallest.
    
    For 3x2x8: min(6, 16, 24) = 6.
    
    Construction of rank-6 decomposition of all-ones tensor (3,2,8):
    We need: sum_{c=0}^{5} U[i*2+j, c] * V[j*8+k, c] * W[k*3+i, c] = 1 for all i,j,k.
    
    One valid construction:
    Let U[i*2+j, c] = 1 for c = 0..5 (all ones in a 6x6 submatrix)
    Let V[j*8+k, c] = 1 for c = 0..5
    Let W[k*3+i, c] = 1 for c = 0..5
    Then sum_c U*V*W = 6 * 1 * 1 = 6, not 1.
    
    So we need to scale by 1/6, but 1/6 is not a half-integer.
    
    Alternative: Use a different construction.
    
    Actually, the standard construction for the all-ones tensor of size n x m x p with rank r = min(nm, mp, pn):
    If r = nm, then set U[i*m+j, c] = delta_{c, i*m+j} (identity), V[j*p+k, c] = 1 for all c, W[k*n+i, c] = 1 for all c.
    Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_c delta_{c,i*m+j} * 1 * 1 = 1.
    
    Wait, that gives rank = nm, not min(nm,mp,pn).
    
    Let me reconsider. The all-ones tensor T[i,j,k] = 1.
    We can write T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c where:
    u_c is a vector in R^{nm}, v_c in R^{mp}, w_c in R^{pn}.
    
    If we set r = nm:
    u_c = e_c (standard basis vector in R^{nm})
    v_c = 1 (all-ones vector in R^{mp})
    w_c = 1 (all-ones vector in R^{pn})
    Then (u_c ⊗ v_c ⊗ w_c)[i*m+j, j*p+k, k*n+i] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = delta_{c, i*m+j} * 1 * 1
    Sum over c: sum_c delta_{c, i*m+j} = 1. Correct!
    
    So rank(n,m,p) <= nm for the all-ones tensor. Similarly rank <= mp and rank <= pn.
    Therefore rank = min(nm, mp, pn).
    
    For 3x2x8: rank = min(6, 16, 24) = 6.
    Construction: r = 6 = nm.
    U[i*2+j, c] = delta_{c, i*2+j} for c in 0..5
    V[j*8+k, c] = 1 for all j,k,c
    W[k*3+i, c] = 1 for all k,i,c
    
    Let me verify: sum_c U[i*2+j,c] V[j*8+k,c] W[k*3+i,c] = sum_c delta_{c,i*2+j} * 1 * 1 = 1. Correct!
    
    So for the full 3x4 @ 4x8:
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    Split into two parts:
    Part 1: j in {0,1}: C1[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    Part 2: j in {2,3}: C2[i,k] = A[i,2]B[2,k] + A[i,3]B[3,k]
    
    For Part 1, this is the 3x2 @ 2x8 all-ones tensor with rank 6.
    For Part 2, this is also the 3x2 @ 2x8 all-ones tensor with rank 6.
    
    Total rank = 12? That seems suspiciously low. Let me double-check.
    
    Wait, I think I'm confusing the tensor structure. Let me be more careful.
    
    The multiplication tensor for 3x4 @ 4x8 is T[i,j,k] where:
    T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    Actually, T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_{j,k} T[i,j,k] A[i,j] B[j,k]
    
    The tensor has indices (i,j,k) with i in 0..2, j in 0..3, k in 0..7.
    T[i,j,k] = 1 for all i,j,k.
    
    This is the all-ones tensor of shape (3,4,8) with rank min(3*4, 4*8, 8*3) = min(12, 32, 24) = 12.
    
    Oh! So the entire 3x4 @ 4x8 multiplication tensor IS the all-ones tensor of shape (3,4,8),
    and its rank is min(12, 32, 24) = 12!
    
    Let me verify the construction:
    r = 12 = n*m = 3*4.
    U[i*4+j, c] = delta_{c, i*4+j} for c in 0..11
    V[j*8+k, c] = 1 for all j,k,c
    W[k*3+i, c] = 1 for all k,i,c
    
    Check: sum_c U[i*4+j, c] V[j*8+k, c] W[k*3+i, c] = sum_c delta_{c, i*4+j} * 1 * 1 = 1. Correct!
    
    So rank = 12!
    
    But wait, is this really correct? Let me think again.
    
    The multiplication tensor T satisfies:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as:
    C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    
    where T[i,j,k] = 1 for all i,j,k.
    
    The tensor rank of T is the minimum r such that:
    T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] V[j,k,c] W[k,i,c]
    
    With the construction above, r = 12 works. And since rank >= max(n, m, p) = 8, and rank <= min(nm, mp, pn) = 12, we have rank in [8, 12]. The construction gives 12.
    
    Actually, I realize the tensor indices need to match the convention. The problem states:
    U has shape (n*m, rank) = (12, rank)
    V has shape (m*p, rank) = (32, rank)  
    W has shape (p*n, rank) = (24, rank)
    
    And the decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] = 1 for the multiplication tensor.
    
    So yes, rank = 12 is achievable!
    """
    n, m, p = 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Use rank = min(nm, mp, pn) = 12
    rank = min(nm, mp, pn)  # 12
    
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    # Construction: U[i*m+j, c] = delta_{c, i*m+j}, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    for c in range(rank):
        i, j = divmod(c, m)
        U[i * m + j][c] = 1.0
        for jj in range(m):
            for kk in range(p):
                V[jj * p + kk][c] = 1.0
        for kk in range(p):
            for ii in range(n):
                W[kk * n + ii][c] = 1.0
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                assert abs(total - 1.0) < 1e-10, f"Verification failed at ({i},{j},{k}): got {total}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
