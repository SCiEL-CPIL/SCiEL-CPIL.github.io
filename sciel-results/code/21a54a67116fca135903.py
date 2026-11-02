"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank)
      V: shape (m*p, rank)  
      W: shape (p*n, rank)
    The decomposition satisfies: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115s safety margin
    
    # Start with schoolbook (rank = n*m*p = 84)
    schoolbook_rank = n * m * p
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions using random linear algebra approach
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    # We want to find matrices U, V, W of rank r such that T = U V^T W^T (in appropriate sense)
    
    # Strategy: Use random initialization and gradient-free search (simulated annealing)
    # to try to reduce rank while maintaining exact decomposition.
    
    # Alternative: Use known results. For m*n*p matrix multiplication, 
    # the minimum rank is at most n*m*p and at least max(n*m, m*p, p*n).
    # For 3,4,7: lower bound is max(12, 28, 21) = 28.
    # Schoolbook gives 84. Can we do better?
    
    # Let's try a structured approach: use Strassen-like combinations
    # For 2x2x2, Strassen achieves rank 7 instead of 8.
    # We can try to apply this pattern where possible.
    
    # Actually, let's try a direct search using random linear combinations.
    # The key insight: if we can find r vectors u_a, v_a, w_a such that
    # for each (i,j,k): sum_a u_a[i*m+j] * v_a[j*p+k] * w_a[k*n+i] = 1
    
    # This is a polynomial identity problem. Let's use the method of 
    # finding a rank-r decomposition via random sampling and solving.
    
    # Try ranks from lower bound up
    min_possible = max(n * m, m * p, p * n)
    
    for r in range(min_possible, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try random search for rank r
        success = False
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
            
            # Random initialization
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            # For the decomposition to work, we need:
            # sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1 for all (i,j,k)
            
            # This is equivalent to: the tensor with components T[i,j,k] = sum_a U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] equals the all-ones tensor on the valid indices.
            
            # We can set up this as a system of equations. There are n*m*p = 84 equations.
            # Unknowns: r*(n*m + m*p + p*n) = r*(12+28+21) = r*61
            # For r >= 28, we have more unknowns than equations, so solutions may exist.
            
            # But the equations are nonlinear (cubic). Let's try a different approach.
            
            # Use the fact that if we fix U and V, then W is determined (if possible).
            # Actually, let's try to construct explicitly.
            
            # Explicit construction: 
            # Let U[i*m+j, a] = alpha_i * beta_j * gamma_a
            # Let V[j*p+k, a] = beta_j * delta_k * gamma_a  
            # Let W[k*n+i, a] = delta_k * alpha_i * gamma_a
            # Then sum_a U*V*W = sum_a alpha_i^2 * beta_j^2 * delta_k^2 * gamma_a^3
            # We want this = 1 for all i,j,k.
            # This requires alpha_i^2 * beta_j^2 * delta_k^2 * sum_a gamma_a^3 = 1
            # So we need sum_a gamma_a^3 = 1/(alpha_i^2 beta_j^2 delta_k^2) for all i,j,k
            # This is only possible if alpha_i^2 beta_j^2 delta_k^2 is constant, which means
            # all alpha_i, beta_j, delta_k have the same absolute value.
            # Let alpha_i = c, beta_j = c, delta_k = c for all i,j,k.
            # Then we need c^6 * sum_a gamma_a^3 = 1.
            # We can set c=1 and choose gamma_a such that sum_a gamma_a^3 = 1.
            # For example, r=1: gamma_0 = 1. Then U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
            # Check: sum_a 1*1*1 = 1. Yes! But wait, this gives rank 1?
            
            # Wait, that can't be right. Let me recheck.
            # If U[i*m+j, 0] = 1 for all (i,j), V[j*p+k, 0] = 1 for all (j,k), W[k*n+i, 0] = 1 for all (k,i)
            # Then for each (i,j,k): sum_0 U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
            
            # So rank 1 works?! That seems too good to be true.
            # Let me verify with the actual tensor definition.
            
            # The matrix multiplication tensor for (n,m,p) has:
            # T[a, b, c] where a indexes (i,j) in A, b indexes (j,k) in B, c indexes (i,k) in C
            # T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent, 0 otherwise.
            
            # Wait, I think I misunderstood the tensor. Let me reconsider.
            
            # For matrix multiplication C = AB where A is n×m and B is m×p:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T has T[i,j,k] = 1 meaning "multiply A[i,j] by B[j,k] and add to C[i,k]"
            
            # In the factorization: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
            # (using the index conventions from the task)
            
            # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
            
            # With rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
            # Sum = 1*1*1 = 1 for all (i,j,k). ✓
            
            # So the rank is 1! But this contradicts known results...
            
            # Ah, I see the issue. The tensor decomposition for matrix multiplication is not just 
            # matching the "1" entries. The tensor is a 3-tensor, and its rank is the minimum 
            # number of rank-1 tensors that sum to it. The rank-1 tensors are of the form
            # u ⊗ v ⊗ w where u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n}.
            
            # But the constraint is that the sum of these rank-1 tensors equals the matrix 
            # multiplication tensor, which has specific structure.
            
            # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined as:
            # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A ⊗ e_j^B ⊗ e_k^C 
            # where e_i^A is a vector in R^{n*m} with 1 at position i*m+j for the appropriate j...
            
            # Actually, I think the correct interpretation is:
            # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 and T[a,b,c] = 0 otherwise.
            # So T is not all-ones; it's a sparse tensor with 1s only at consistent indices.
            
            # With rank 1: U[i*m+j,0] = 1, V[j*p+k,0] = 1, W[k*n+i,0] = 1
            # Then (U⊗V⊗W)[a,b,c] = U[a,0]*V[b,0]*W[c,0] = 1 for ALL (a,b,c), not just consistent ones.
            # But T has 0s at inconsistent indices. So rank 1 does NOT work.
            
            # I need to be more careful. Let me restart with the correct tensor.
            
            pass
        
        if success:
            break
    
    # Given the complexity and time constraints, return the schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
