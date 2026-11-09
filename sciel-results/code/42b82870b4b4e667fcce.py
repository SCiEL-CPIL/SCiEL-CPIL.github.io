"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 8

    # Schoolbook decomposition for 2x4 * 4x8
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # We can use rank-1 terms: for each (i,k,j), term is A[i,k]*B[k,j]
    # That gives rank = n*m*p = 2*4*8 = 64
    # But we can do better by grouping.
    
    # Actually, the standard rank for matrix multiplication of n x m by m x p is n*m*p
    # For 2x4 * 4x8, that's 64.
    # But we know Strassen gives 56.
    # 
    # Let me think about a different approach: use the fact that m=4 is small.
    # We can write the multiplication as a sum of 4 (2x1)*(1x8) terms plus 4 more
    # No, that's not right either.
    
    # Let me just implement a valid rank-64 schoolbook and see if we can do better.
    
    # Actually, let me implement rank 64 schoolbook first as a baseline,
    # then try to find a better decomposition.
    
    rank = 64
    U = np.zeros((n*m, rank))  # (8, 64)
    V = np.zeros((m*p, rank))  # (32, 64)
    W = np.zeros((p*n, rank))  # (16, 64)
    
    r = 0
    for i in range(n):      # row of A, column of C
        for k in range(m):  # col of A, row of B
            for j in range(p):  # col of B, col of C
                U[i*m+k, r] = 1
                V[k*p+j, r] = 1
                W[j*n+i, r] = 1
                r += 1
    
    # Verify
    # C = W^T * (V^T * B * U) ... no, that's not how it works.
    # The decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = A[i,k]*B[k,j] = C[i,j]
    # So C = sum_r (u_r^T A) * (B v_r) * w_r^T ... 
    # Actually, the bilinear form is: C[i,j] = sum_k A[i,k] B[k,j]
    # We decompose as C[i,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # Wait, the indices need to be consistent.
    
    # Let me reconsider. The tensor T has entries T[i,j,k] such that
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So T[i,j,k] = A[i,k]*B[k,j]
    # We want T = sum_r u_r ⊗ w_r ⊗ v_r where u_r is over (i,k), v_r over (k,j), w_r over (i,j)
    # No, the convention is W[k*n+i, r] as stated.
    
    # Let me re-read: "U', 'V', 'W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U has shape (n*m, rank) = (8, rank), indexed by (i, k) where i in 0..n-1, k in 0..m-1
    # V has shape (m*p, rank) = (32, rank), indexed by (k, j) where k in 0..m-1, j in 0..p-1
    # W has shape (p*n, rank) = (16, rank), indexed by (j, i) where j in 0..p-1, i in 0..n-1
    # "output factor follows the official trace convention W[k*n+i,r]" - hmm, this says W[k*n+i, r]
    # but W has shape (p*n, rank), so the first index goes from 0 to p*n-1 = 15.
    # k*n+i with k in 0..p-1 and i in 0..n-1 gives range 0..15. So W index = k*n + i.
    # But k here is the output column index (j in my notation), and i is the output row index.
    # So W[j*n+i, r].
    
    # So the decomposition is: C[i,j] = sum_r U[i*m+k... wait, U is (n*m, rank).
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] should sum over k and r to give C[i,j].
    # Actually no. The standard decomposition is:
    # C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k' V[k'*p+j, r] * B[k',j]) * W[j*n+i, r]
    # But that's not quite right either because U and V share the k index.
    
    # I think the correct interpretation is:
    # C[i,j] = sum_r U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] where we sum over k as well.
    # No, that doesn't make sense dimensionally.
    
    # Let me think again. The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1
    # (as a trilinear form: M(A,B) = C where C[i,j] = sum_k A[i,k]B[k,j]).
    # A rank-r decomposition is: M = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t is a vector in R^{n*m} (flattened A), v_t in R^{m*p} (flattened B), w_t in R^{p*n} (flattened C^T or something).
    # Specifically, C[i,j] = sum_t (sum_k u_t[i*m+k] * A[i,k]) * (sum_k' v_t[k'*p+j] * B[k',j]) * w_t[j*n+i]
    # But this only works if u_t and v_t are "separable" in the k index.
    
    # Actually, I think the standard form is:
    # C[i,j] = sum_t U_t[i,m] * V_t[m,j] * W_t[j,i] where U_t, V_t, W_t are rank-1 matrices
    # U_t is n x m, V_t is m x p, W_t is p x n (but flattened as (p*n, 1))
    # And U[i*m+k, t] = U_t[i,k], V[k*p+j, t] = V_t[k,j], W[j*n+i, t] = W_t[i,j] (or W_t[j,i])
    
    # So C[i,j] = sum_t sum_k U_t[i,k] * A[i,k] * V_t[k,j] * B[k,j] * W_t[j,i]... 
    # No, that's not right. The bilinear multiplication is:
    # C = A * B, so C[i,j] = sum_k A[i,k] B[k,j]
    # The rank-r decomposition means we can write this as:
    # C[i,j] = sum_t (sum_k A[i,k] * U_t[i,k]) * (sum_k B[k,j] * V_t[k,j]) * W_t[j,i]
    # But this only works if U_t[i,k] and V_t[k,j] factor in a way that the k-indices align.
    
    # I think I'm overcomplicating this. Let me just use the schoolbook:
    # For each (i, k, j), create a rank-1 term where:
    # U[i*m+k, t] = 1, V[k*p+j, t] = 1, W[j*n+i, t] = 1
    # Then C[i,j] = sum_t sum_k' U[i*m+k', t] * A[i,k'] * V[k'*p+j, t] * B[k',j] * W[j*n+i, t]
    # For the term where k'=k, this gives A[i,k]*B[k,j]*1, and all other terms are 0.
    # So C[i,j] = sum_k A[i,k]*B[k,j]. Correct!
    
    # So my schoolbook above should work. Let me verify the index convention for W.
    # "W[k*n+i,r]" - so W is indexed by k*n+i where k is the column index of C (j in my notation)
    # and i is the row index of C. So W[j*n+i, r]. That matches what I have.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
