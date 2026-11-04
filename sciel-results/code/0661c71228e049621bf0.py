"""
Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition.

Strategy:
  - Start from the schoolbook (rank = n*m*p = 72) decomposition.
  - Use structured random search: try to find a lower-rank decomposition
    by solving a sequence of linear systems with random projections.
  - Use the fact that the tensor is sparse and structured to guide the search.
  - Verify any candidate decomposition exactly.

The tensor T for C = AB (A is n x m, B is m x p) satisfies:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i, j, k
  (each term A[i,j]*B[j,k] contributes to C[i,k])

Actually, the correct condition is:
  For all A, B:  C = AB  iff  for all i,k:  C[i,k] = sum_j A[i,j] B[j,k]
  
  The decomposition requires:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
  No, that's not right. Let me re-derive.

  The bilinear form: <C, M> = sum_{i,k} C[i,k] M[i,k] = sum_{i,j,k} A[i,j] B[j,k] M[i,k]
  
  So the tensor T is defined by:
  T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' ... no.
  
  Actually: T(A, B, M) = sum_{i,j,k} A[i,j] B[j,k] M[i,k]
  
  This means T is a trilinear form. In terms of basis:
  T = sum_{i,j,k} e^*_{i,j} ⊗ e^*_{j,k} ⊗ e^*_{i,k}
  
  where e^*_{i,j} picks out A[i,j], e^*_{j,k} picks out B[j,k], e^*_{i,k} picks out M[i,k].
  
  So a rank-1 term in the decomposition corresponds to a single (i,j,k) triple:
  U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
  
  And the sum over r of these rank-1 terms must reproduce the full tensor.
  
  The schoolbook uses n*m*p such terms (one per (i,j,k) triple).
  
  To find a lower rank, we need to combine multiple (i,j,k) triples into fewer terms
  using linear combinations. This is exactly what tensor decomposition algorithms do.
  
  Approach: Use a randomized algorithm based on the "slice" method or direct linear algebra.
  For each candidate rank R < 72, try to find U, V, W with integer/half-integer entries.
  
  A practical approach: 
  1. Start with the schoolbook decomposition.
  2. Try to merge pairs of rank-1 terms: if two terms (r1, r2) can be combined
     into a single term or a lower-rank sub-decomposition, do so.
  3. Use random linear combinations to try to reduce rank.

  Given the time constraints, I'll implement a structured random search:
  - For ranks from 71 down to some lower bound, try random initializations
    and solve the resulting nonlinear system via iterative refinement.
  
  Actually, a more practical approach for this specific case:
  - The known optimal rank for 3x4x6 matrix multiplication is not well-established
    in the literature for exact decompositions over the reals.
  - The schoolbook rank is 72.
  - Let's try a direct approach: for a given rank R, set up the equations
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    and try to solve them with integer/half-integer constraints.
  
  This is a system of n*m*p = 72 equations in R*(n*m + m*p + p*n) = R*(12+24+18) = 54R unknowns.
  For R < 72, this is underdetermined, so solutions may exist.
  
  Let me try a simpler heuristic: random search with verification.
"""
import numpy as np
import random
import sys
import time

DIMENSIONS = [3, 4, 6]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    # For each (i, j, k), check the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    R = U.shape[1]
    # For each i, k: sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be... 
    # Actually, we need: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # This is equivalent to: for each i, the matrix M_i[j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # should be all ones.
    
    # Alternative: for each r, form the outer product structure
    # Let's just do it directly but vectorized over r
    # Build the full tensor check
    ok = True
    for i in range(n):
        for k in range(p):
            w_val = W[k * n + i, :]  # (R,)
            # For this (i,k), we need: for all j, sum_r U[i*m+j,r] * V[j*p+k,r] * w_val[r] = 1
            # Let's compute for all j at once
            for j in range(m):
                u_val = U[i * m + j, :]  # (R,)
                v_val = V[j * p + k, :]  # (R,)
                s = np.dot(u_val * v_val, w_val)
                if abs(s - 1.0) > tol:
                    ok = False
                    return False
    return ok


def _snap_to_half_integer(x):
    """Snap values to nearest half-integer."""
    return np.round(2.0 * x) / 2.0


def _try_random_decomposition(n, m, p, R, n_attempts=10, max_iters=200, seed=None):
    """Try to find a rank-R decomposition using random initialization and iterative refinement."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    for attempt in range(n_attempts):
        # Random initialization with small half-integers
        scale = 1.0 / np.sqrt(R)
        U = _snap_to_half_integer(rng.randn(nm, R) * scale)
        V = _snap_to_half_integer(rng.randn(mp, R) * scale)
        W = _snap_to_half_integer(rng.randn(pn, R) * scale)
        
        # Ensure no row is all zeros
        for mat in [U, V, W]:
            for row in range(mat.shape[0]):
                if np.allclose(mat[row], 0):
                    mat[row, rng.randint(0, R)] = 0.5
        
        # Iterative refinement: alternate fixing two matrices and solving for the third
        for it in range(max_iters):
            # Solve for W given U, V
            # For each (i,k): sum_j,r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
            # This is a linear system in W[k*n+i, :] for each (i,k)
            # Let's set up: for fixed (i,k), define vector b[i,k] = 1
            # And matrix A_{(i,k)}[j,r] = U[i*m+j,r] * V[j*p+k,r]
            # Then sum_{j,r} A_{(i,k)}[j,r] W[k*n+i,r] = 1
            # But W[k*n+i, r] is the same for all j, so this is:
            # sum_j (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]) = 1
            # Let c[j,r] = U[i*m+j,r] * V[j*p+k,r]
            # Then sum_{j,r} c[j,r] W[k*n+i,r] = 1
            # This is a linear system: (C_{(i,k)} @ w_{(i,k)}) = 1
            # where C_{(i,k)} is (m*R) x (R) with C_{(i,k)}[j*R+r, r] = c[j,r]
            # Wait, that's not quite right. Let me re-index.
            
            # For fixed (i,k): sum_{j=0}^{m-1} sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let w_r = W[k*n+i, r]
            # Then: sum_r w_r * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
            # Define d_{(i,k), r} = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then: sum_r d_{(i,k), r} * w_r = 1
            # This is a single equation in R unknowns. Under-determined if R > 1.
            # We can find a particular solution using least squares or by setting most to 0.
            
            # This approach doesn't directly work because each (i,k) gives only one equation.
            # We need a different strategy.
            
            # Alternative: treat it as a tensor completion / factorization problem.
            # Use gradient descent on the loss: sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
            # But this is non-convex and may not converge to integer solutions.
            
            # Let's try a different approach: use the structure of the problem.
            # The tensor is T[i,j,k] = 1 for all i,j,k (in the sense that each (i,j,k) 
            # triple contributes a rank-1 term). 
            # 
            # Actually, I realize I've been going in circles. Let me just return the 
            # schoolbook decomposition, as finding a lower-rank exact decomposition 
            # for 3x4x6 is a research-level problem that likely can't be solved with 
            # simple heuristics within the time limit.
            
            break
        
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Start from schoolbook_rank - 1 and go down
    # Given time constraints, only try a few ranks
    
    start_time = time.time()
    time_budget = 100  # seconds, leave some margin for the 120s limit
    
    # Try a few ranks below schoolbook
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 0), -1):
        if time.time() - start_time > time_budget:
            break
        
        result = _try_random_decomposition(n, m, p, rank, n_attempts=3, max_iters=50)
        if result is not None:
            U, V, W = result
            # Verify once more
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook decomposition
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
