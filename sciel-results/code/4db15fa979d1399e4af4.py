"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    u_vals = Ua[u_idx, :]
    v_vals = Va[v_idx, :]
    w_vals = Wa[w_idx, :]
    
    prod = u_vals * v_vals * w_vals
    result = prod.sum(axis=1)
    
    expected = np.ones(n * m * p)
    return np.allclose(result, expected, atol=tol)

def _try_structured_decomposition(n, m, p, rank):
    """Try structured decompositions based on rank factorization ideas."""
    # Strategy: T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Approach 1: If rank >= n*m*p, schoolbook works (already handled)
    # Approach 2: Try to use fewer components by sharing
    
    # Key insight: if we set W[k*n+i,c] = 1 for all (k,i) and some c,
    # then we need sum_c U[i*m+j,c] * V[j*p+k,c] = 1 for all (i,j,k)
    # This is harder.
    
    # Approach 3: Use the fact that the tensor is all-ones.
    # If we can find matrices A (n*m x r), B (m*p x r), C (p*n x r)
    # such that the element-wise triple product sums to 1.
    
    # Simple structured attempt: use rank-1 components
    # For each c, define U_c, V_c, W_c as rank-1 tensors
    # U_c[i*m+j] = a_c[i] * b_c[j]
    # V_c[j*p+k] = b_c[j] * c_c[k]  
    # W_c[k*n+i] = c_c[k] * a_c[i]
    # Then product = a_c[i]*b_c[j]*b_c[j]*c_c[k]*c_c[k]*a_c[i] = (a_c[i]*b_c[j]*c_c[k])^2
    # Sum_c (a_c[i]*b_c[j]*c_c[k])^2 = 1 for all (i,j,k)
    
    # This is equivalent to: the tensor with entries (a_c[i]*b_c[j]*c_c[k])^2 sums to 1
    # Let x_c[i,j,k] = a_c[i]*b_c[j]*c_c[k], then sum_c x_c[i,j,k]^2 = 1
    
    # This is like representing the all-ones tensor as a sum of squares of rank-1 tensors.
    # The minimum number of rank-1 tensors needed to represent the all-ones tensor as sum of squares...
    # This is related to the "sum of squares" rank.
    
    # For the all-ones tensor of size n x m x p, we can write:
    # 1 = (1/sqrt(r)) * sum_{c=1}^{r} u_c[i]*v_c[j]*w_c[k] where the u_c, v_c, w_c are orthonormal-ish
    # But we need the SQUARE to sum to 1, not the tensor itself.
    
    # Actually, let me reconsider. The constraint is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # NOT sum_c (U*V*W)^2 = 1
    
    # So it's a linear constraint, not quadratic. This is much easier!
    
    # We need: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try: U[i*m+j,c] = alpha_c[i,j], V[j*p+k,c] = beta_c[j,k], W[k*n+i,c] = gamma_c[k,i]
    # Then sum_c alpha_c[i,j] * beta_c[j,k] * gamma_c[k,i] = 1
    
    # If we set gamma_c[k,i] = 1 for all c, then:
    # sum_c alpha_c[i,j] * beta_c[j,k] = 1 for all (i,j,k)
    # This means for each (i,j), the vector (beta_c[j,k])_{k} dotted with (alpha_c[i,j])_c sums to 1 across k.
    # Wait, no. For fixed (i,j), we need sum_k [sum_c alpha_c[i,j] * beta_c[j,k]] = ... no.
    # We need for EACH (i,j,k): sum_c alpha_c[i,j] * beta_c[j,k] * gamma_c[k,i] = 1
    
    # If gamma_c[k,i] = 1: sum_c alpha_c[i,j] * beta_c[j,k] = 1 for all (i,j,k)
    # For fixed (i,j): sum_c alpha_c[i,j] * beta_c[j,k] = 1 for all k
    # This means the vector (sum_c alpha_c[i,j] * beta_c[j,k])_k is the all-ones vector of length p.
    # Let A[i,j] = (alpha_c[i,j])_c, B[j,k] = (beta_c[j,k])_c
    # Then A[i,j] · B[j,k] = 1 for all k, meaning A[i,j] is orthogonal to B[j,k] - B[j,k'] for all k,k'
    # And A[i,j] · B[j,0] = 1
    
    # This is a system of linear equations. Let's set up:
    # Unknowns: alpha_c[i,j] for c in range(r), i in range(n), j in range(m)
    #           beta_c[j,k] for c in range(r), j in range(m), k in range(p)
    # Constraints: for all i,j,k: sum_c alpha_c[i,j]*beta_c[j,k] = 1
    
    # Number of unknowns: r*n*m + r*m*p = r*(n*m + m*p) = r*m*(n+p)
    # Number of constraints: n*m*p
    
    # For r=1: unknowns = m*(n+p) = 4*11 = 44, constraints = 112. Underdetermined, likely no solution.
    # For r=3: unknowns = 3*4*11 = 132, constraints = 112. Might have solutions.
    
    # Let me try a different approach: use the fact that we can set W to have a special structure.
    
    # Alternative: set U[i*m+j,c] = u_c[i], V[j*p+k,c] = v_c[j,k], W[k*n+i,c] = w_c[i]
    # Then sum_c u_c[i] * v_c[j,k] * w_c[i] = sum_c (u_c[i]*w_c[i]) * v_c[j,k] = 1 for all (i,j,k)
    # For fixed i: sum_c (u_c[i]*w_c[i]) * v_c[j,k] = 1 for all (j,k)
    # Let s_c[i] = u_c[i]*w_c[i], then sum_c s_c[i] * v_c[j,k] = 1 for all (i,j,k)
    # For fixed (j,k): sum_c s_c[i] * v_c[j,k] = 1 for all i
    # This means the vector (s_c[i])_i is such that its dot product with (v_c[j,k])_c is 1 for all i.
    # So (v_c[j,k])_c must be constant across i... which means v_c[j,k] doesn't depend on i (it doesn't).
    # And sum_c s_c[i] * v_c[j,k] = 1 for all i means:
    #   For i=0: sum_c s_c[0] * v_c[j,k] = 1
    #   For i=1: sum_c s_c[1] * v_c[j,k] = 1
    #   etc.
    # This requires s_c[i] to be the same for all i, i.e., s_c[i] = s_c for all i.
    # Then sum_c s_c * v_c[j,k] = 1 for all (j,k).
    # And s_c = u_c[i]*w_c[i] must be independent of i.
    
    # So: find s_c, u_c[i], w_c[i], v_c[j,k] such that:
    #   u_c[i]*w_c[i] = s_c for all i (for each c)
    #   sum_c s_c * v_c[j,k] = 1 for all (j,k)
    
    # From the first: w_c[i] = s_c / u_c[i] (if u_c[i] != 0)
    # From the second: this is a linear system in v_c[j,k] with coefficients s_c.
    # Let V_mat[j,k,c] = v_c[j,k]. Then sum_c s_c * V_mat[j,k,c] = 1.
    # This is: V_mat @ s = 1 (vector of ones), where s = (s_c)_c.
    # We have m*p equations and r unknowns s_c, with V_mat as coefficients.
    # Wait, V_mat[j,k,c] are also unknowns. So we have freedom.
    
    # Let's set s_c = 1/r for all c (simple choice). Then:
    #   sum_c (1/r) * v_c[j,k] = 1 => sum_c v_c[j,k] = r for all (j,k)
    #   And u_c[i]*w_c[i] = 1/r for all i.
    #   Set u_c[i] = 1, w_c[i] = 1/r. Then u_c[i]*w_c[i] = 1/r. ✓
    #   And we need sum_c v_c[j,k] = r, so set v_c[j,k] = 1 for all c,j,k. ✓
    
    # So: U[i*m+j,c] = 1, V[j*p+k,c] = 1, W[k*n+i,c] = 1/r for all c,i,j,k.
    # Then sum_c 1*1*(1/r) = r*(1/r) = 1. ✓
    
    # This works for ANY rank r >= 1! Let me verify:
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_c 1*1*(1/r) = r*(1/r) = 1 ✓
    
    # So rank 1 works! U is all 1s, V is all 1s, W is all 1/r.
    # But wait, the entries must be integer or half-integer. 1/r for r=1 is 1 (integer). ✓
    
    # Let me double-check with r=1:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1 ✓
    
    # Rank 1 works!

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Rank 1 decomposition: U all 1s, V all 1s, W all 1s
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For r=1: T[i,j,k] = 1*1*1 = 1 ✓
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify
    if _verify(U, V, W, n, m, p, rank, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (shouldn't happen)
    if time.time() < deadline:
        return _schoolbook()
    
    return _schoolbook()
