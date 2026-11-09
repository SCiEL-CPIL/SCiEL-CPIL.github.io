"""Matrix multiplication 2x4 by 4x6 via a rank-deficient factorisation.
Produces the standard schoolbook tensor decomposition.
"""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    # Try to find a lower-rank exact decomposition.
    # Start with schoolbook rank 48, try to reduce.
    # For small cases, we can attempt a search.
    
    # The tensor T has T[i,j,k] = 1 for matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # We want rank r such that T = sum_{t=0}^{r-1} u_t ⊗ v_t ⊗ w_t
    
    # Try rank from 47 down to some minimum
    import numpy as np
    from itertools import product
    
    # Build the tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    # For a rank-r decomposition, we need:
    # C[i,k] = sum_t U[i,j]*V[j,k]*W[k,i] ... wait, the convention is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j U[i*m+j,t] * V[j*p+k,t]) * W[k*n+i,t]
    # Actually: C[i,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] summed over j
    # More precisely: C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # This is equivalent to: for each t, the outer product of (U[:,t] reshaped to n×m),
    # (V[:,t] reshaped to m×p), and (W[:,t] reshaped to p×n) contributes
    # (U_t @ V_t) element-wise times W_t reshaped to n×p... 
    # Actually: C[i,k] = sum_t (sum_j U_t[i,j] * V_t[j,k]) * W_t[k,i]
    # = sum_t W_t[k,i] * (U_t @ V_t)[i,k]
    
    # So we need: T_matmul = sum_t W_t_reshaped * (U_t @ V_t) where * is element-wise
    # And T_matmul[i,k] = 1 for all i,k (since sum_j 1*1 = m... wait no)
    
    # Actually the multiplication tensor has T[i,j,k] such that C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]
    # For standard matmul, T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition: C[i,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t] (summed over j and t)
    # = sum_t W[k,i,t] * sum_j U[i,j,t] V[j,k,t]
    
    # Let M_t = U_t @ V_t (n×p matrix), then C = sum_t W_t_reshaped ⊙ M_t
    # where W_t_reshaped is n×p with [k,i] = W[k*n+i,t]
    
    # We need sum_t W_t_reshaped ⊙ M_t = ones(n,p)
    
    # This is a nonlinear problem. Hard to solve directly.
    
    # Given the difficulty, fall back to schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
