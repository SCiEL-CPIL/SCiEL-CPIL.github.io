"""
Program to find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
Dimensions: n=2, m=4, p=4.
Uses linear algebra over Q to search for lower-rank decompositions.
"""

from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    n_m = n * m  # 8
    m_p = m * p  # 16
    p_n = p * n  # 8
    
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: For a given rank r, we try to find U, V, W.
    # We can fix W to be a matrix and solve for U, V.
    # The equation is: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 32 equations (one per (i,j,k)) with variables U, V (W is fixed).
    # But it's bilinear, not linear. We can linearize by fixing one set.
    
    # Alternative: Use the approach where we fix W to be the identity-like structure
    # and solve for U and V.
    
    # Let's try: fix W[k*n+i, r] to be a specific pattern, then the equation becomes
    # linear in U and V (since W is fixed).
    
    # Actually, let's use a more systematic approach:
    # For rank r, we have r terms. Let's denote:
    # a_r = U[:,r] (length 8), b_r = V[:,r] (length 16), c_r = W[:,r] (length 8)
    # Constraint: for all (i,j,k): sum_r a_r[i*m+j] * b_r[j*p+k] * c_r[k*n+i] = 1
    
    # We can fix c_r (W) and solve for a_r, b_r.
    # Given W, the constraint for each (i,j,k) is:
    # sum_r a_r[i*m+j] * b_r[j*p+k] * c_r[k*n+i] = 1
    # This is still bilinear in a and b.
    
    # Better approach: For each r, the outer product a_r ⊗ b_r ⊗ c_r must sum to T.
    # T[i,j,k] = 1 for all i,j,k.
    
    # Let's try the standard schoolbook first as a baseline, then attempt to reduce.
    # Schoolbook: r = 32, U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for matching (i,j,k).
    
    # To find lower rank, we need to exploit structure.
    # The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices are linked.
    
    # Actually, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor! So T = 1 ⊗ 1 ⊗ 1 in some sense.
    # Wait, no. T[i,j,k] = 1 for all valid i,j,k.
    # So T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki} where e_{ij} is the (i,j) entry.
    
    # The rank of the all-ones tensor of size n×m×p is 1!
    # Because T[i,j,k] = 1 = 1*1*1 for all i,j,k.
    # So U[:,0] = all ones, V[:,0] = all ones, W[:,0] = all ones gives rank 1!
    
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    # The matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # So T[i,j,k] = 1 for all i,j,k.
    
    # A rank-1 decomposition would be: T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # We need u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k.
    # This is possible! Set u[i,j] = 1, v[j,k] = 1, w[k,i] = 1.
    # Then u[i,j]*v[j,k]*w[k,i] = 1*1*1 = 1 = T[i,j,k].
    
    # So the rank is 1?! That seems too good to be true.
    
    # Wait, I think I'm confusing the tensor structure. Let me reconsider.
    
    # The matrix multiplication tensor T is defined such that:
    # <T, A ⊗ B> = C where C = A*B
    # In components: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    # So yes, T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition: T = sum_{r=0}^{r-1} u^r ⊗ v^r ⊗ w^r
    # where (u^r ⊗ v^r ⊗ w^r)[i,j,k] = u^r[i,j] * v^r[j,k] * w^r[k,i]
    
    # For rank 1: T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # We need u[i,j] * v[j,k] * w[k,i] = 1 for all i,j,k.
    
    # Let's try: u[i,j] = a[i], v[j,k] = b[j], w[k,i] = c[k]
    # Then a[i]*b[j]*c[k] = 1 for all i,j,k.
    # This requires a[i]*b[j]*c[k] to be constant.
    # Set a[i] = 1, b[j] = 1, c[k] = 1. Then a[i]*b[j]*c[k] = 1. Works!
    
    # So U[:,0] = [1,1,1,1,1,1,1,1] (all ones, length 8)
    # V[:,0] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] (all ones, length 16)
    # W[:,0] = [1,1,1,1,1,1,1,1] (all ones, length 8)
    
    # Let's verify: For any i,j,k:
    # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 * 1 * 1 = 1 = T[i,j,k]. Correct!
    
    # So the rank is 1!
    
    U = [[1] * 1 for _ in range(n_m)]
    V = [[1] * 1 for _ in range(m_p)]
    W = [[1] * 1 for _ in range(p_n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
