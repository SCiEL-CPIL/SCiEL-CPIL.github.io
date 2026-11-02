"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=5, p=5):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), check T[i,j,k] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_reduction(n=2, m=5, p=5, max_rank=None, deadline=None):
    """
    Attempt to find a lower-rank decomposition by exploiting structure.
    
    The 2x5x5 matrix multiplication tensor T[i,j,k] = delta_{i, i'} where
    (A is 2x5, B is 5x5, C = AB is 2x5).
    
    T[i,j,k] = 1 if and only if the (i,k) entry of AB is B[j,k]*A[i,j] summed...
    Actually, T[i,j,k] is the coefficient in the bilinear form:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has entries T[i,j,k] = 1 for all valid (i,j,k) — wait, no.
    The matrix multiplication tensor for n x m times m x p has entries:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    Wait, that's not right either. The tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    The structure tensor has T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p).
    That would make it the all-ones tensor, which has rank 1. That's clearly wrong.
    
    Let me reconsider. The matrix multiplication tensor T_{n,m,p} is a trilinear form:
    <T, A, B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    
    where C = AB. So T[i,j,k] = 1 for all (i,j,k). But that gives rank 1, which is absurd.
    
    I think the correct interpretation is that the tensor decomposes as:
    T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    and we need T[i,j,k] to be the "matrix multiplication tensor" which is:
    T[i,j,k] = 1 if the indices are "compatible" in the multiplication sense.
    
    Actually, for standard matrix multiplication, the structure tensor is:
    T[i,j,k] = 1 for all i, j, k (since every (i,j) of A and (j,k) of B contributes to (i,k) of C).
    
    Hmm, but that can't be right because then the rank would be 1.
    
    Let me look at this more carefully. The matrix multiplication tensor T_{n,m,p} is defined by:
    T(e_i, e_j, e_k) = 1 if and only if... 
    
    Actually, I think the standard definition is:
    T_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    
    No wait, that's the all-ones tensor.
    
    The correct matrix multiplication tensor is:
    T_{n,m,p} = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    
    where e_{(i,j)} is the basis vector for the (i,j) entry of A, e_{(j,k)} for B, and e_{(i,k)} for C.
    
    So T[i*m+j, j*p+k, i*p+k] = 1 (using appropriate indexings).
    
    In the convention of this problem:
    U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    W has shape (p*n, rank), indexed by (k,i) -> k*n+i  (note: W[k*n+i, r])
    
    So the tensor entry T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid (i,j,k).
    
    Wait, but that means for each rank-1 term c, we have:
    sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all (i,j,k)
    
    This is the standard matrix multiplication tensor, and its rank is what we're trying to minimize.
    """
    if max_rank is None:
        max_rank = n * m * p - 1
    
    start_time = time.time()
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    # We want to find rank r such that T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n.
    
    # This is equivalent to: for all (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Strategy: Use the fact that n=2 is small. Fix i=0 and i=1.
    # For i=0: sum_c U[j,c] V[j*p+k,c] W[k,c] = 1 for all j,k
    # For i=1: sum_c U[5+j,c] V[j*p+k,c] W[5+k,c] = 1 for all j,k
    
    # Let's denote for each c:
    #   a_c[j] = U[j, c] for j=0..4 (i=0 part)
    #   b_c[j] = U[5+j, c] for j=0..4 (i=1 part)
    #   v_c[j,k] = V[j*p+k, c] = V[j*5+k, c]
    #   w0_c[k] = W[k, c] for k=0..4 (i=0 part of W)
    #   w1_c[k] = W[5+k, c] for k=0..4 (i=1 part of W)
    
    # Constraints:
    # sum_c a_c[j] * v_c[j,k] * w0_c[k] = 1 for all j,k in 0..4
    # sum_c b_c[j] * v_c[j,k] * w1_c[k] = 1 for all j,k in 0..4
    
    # This is a bilinear system. Let's try small ranks.
    
    # For rank r, we have r * (10 + 25 + 10) = r * 45 unknowns, and 50 equations.
    # For r=2: 90 unknowns, 50 equations — overdetermined but might work
    # For r=1: 45 unknowns, 50 equations — likely no solution
    
    # Let's try to solve for small ranks using random initialization and gradient descent
    # over the reals, then check if a rational solution exists nearby.
    
    best_result = None
    
    for r in range(1, min(max_rank + 1, 10)):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if deadline is not None and time.time() > deadline - 30:
                break
            
            # Initialize with random small values
            U_mat = np.random.uniform(-1, 1, (n*m, r))
            V_mat = np.random.uniform(-1, 1, (m*p, r))
            W_mat = np.random.uniform(-1, 1, (p*n, r))
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if deadline is not None and time.time() > deadline - 30:
                    break
                
                # Compute gradients
                # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                # dT/dU[i*m+j,c] = V[j*p+k,c] W[k*n+i,c]
                # dT/dV[j*p+k,c] = U[i*m+j,c] W[k*n+i,c]
                # dT/dW[k*n+i,c] = U[i*m+j,c] V[j*p+k,c]
                
                grad_U = np.zeros_like(U_mat)
                grad_V = np.zeros_like(V_mat)
                grad_W = np.zeros_like(W_mat)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                grad_U[i*m+j, c] += V_mat[j*p+k, c] * W_mat[k*n+i, c]
                                grad_V[j*p+k, c] += U_mat[i*m+j, c] * W_mat[k*n+i, c]
                                grad_W[k*n+i, c] += U_mat[i*m+j, c] * V_mat[j*p+k, c]
                
                U_mat -= lr * grad_U
                V_mat -= lr * grad_V
                W_mat -= lr * grad_W
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c] for c in range(r))
                            err += (s - 1)**2
                if err < 1e-8:
                    # Found a solution! Convert to fractions
                    U_frac = [[Fraction(U_mat[i, c]).limit_denominator(1000000) for c in range(r)] for i in range(n*m)]
                    V_frac = [[Fraction(V_mat[i, c]).limit_denominator(1000000) for c in range(r)] for i in range(m*p)]
                    W_frac = [[Fraction(W_mat[i, c]).limit_denominator(1000000) for c in range(r)] for i in range(p*n)]
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(r))
                                if s != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # First, try to find a lower-rank decomposition
    result = _try_rank_reduction(n, m, p, max_rank=15, deadline=deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
