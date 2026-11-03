"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need rank-r decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Start with schoolbook (rank 36) as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U_sb[i*m+j, c] = 1
                V_sb[j*p+k, c] = 1
                W_sb[k*n+i, c] = 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # The tensor is all-ones, so we're looking for structured low-rank factorizations
    
    # For the all-ones tensor of shape (n*m, m*p, p*n) with the specific index structure,
    # we can exploit the separability.
    
    # Key insight: T[i,j,k] = 1 can be written as product of three vectors if we can
    # factor the index space appropriately.
    
    # Let's try rank = n*m + m*p + p*n - n - m - p + 1 type constructions
    # Or use known results: for matrix multiplication, the border rank and rank
    
    # Actually, for the all-ones tensor, we can use:
    # 1 = sum over some structured decomposition
    
    # Try: use the fact that 1 = (1)(1)(1) but we need to handle the index coupling.
    
    # The constraint is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # where U is indexed by (i,j), V by (j,k), W by (k,i)
    
    # This is equivalent to: the 3-tensor with all entries 1 has rank r.
    
    # For the all-ones tensor of dimensions (a,b,c), the rank is min(a,b,c) 
    # if we allow arbitrary entries, but here we have the specific index structure.
    
    # Let me try a different approach: use linear algebra to find minimum rank.
    
    # The tensor T has entries T[i,j,k]=1. We want to write T as sum of r rank-1 tensors.
    # A rank-1 tensor is u_i * v_j * w_k.
    
    # So we need: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
    
    # Let's try small ranks and see if we can solve this.
    
    # For rank 1: u[i*m+j]*v[j*p+k]*w[k*n+i] = 1 for all i,j,k
    # This requires u[i*m+j] = a_i * b_j, v[j*p+k] = c_j * d_k, w[k*n+i] = e_k * f_i
    # Then a_i*b_j*c_j*d_k*e_k*f_i = 1 for all i,j,k
    # => (a_i*f_i) * (b_j*c_j) * (d_k*e_k) = 1 for all i,j,k
    # This is possible! Set a_i*f_i = 1, b_j*c_j = 1, d_k*e_k = 1
    # So rank 1 might work? Let me check...
    
    # u[i*m+j] = a_i * b_j, v[j*p+k] = c_j * d_k, w[k*n+i] = e_k * f_i
    # Product: a_i*b_j*c_j*d_k*e_k*f_i = (a_i*f_i)(b_j*c_j)(d_k*e_k)
    # If we set a_i=1, f_i=1, b_j=1, c_j=1, d_k=1, e_k=1, then product = 1. 
    # So U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1 for all entries, rank 1!
    
    # Wait, that's just the all-ones vector in each matrix. Let me verify:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] with rank 1:
    # U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1. Yes!
    
    # So rank 1 works! The all-ones decomposition.
    
    U1 = np.ones((n*m, 1))
    V1 = np.ones((m*p, 1))
    W1 = np.ones((p*n, 1))
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = U1[i*m+j, 0] * V1[j*p+k, 0] * W1[k*n+i, 0]
                assert abs(val - 1) < 1e-10, f"Failed at {i},{j},{k}: {val}"
    
    return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
# EVOLVE-BLOCK-END
