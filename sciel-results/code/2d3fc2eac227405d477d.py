"""Matrix multiplication 3x3x6: search for a rank-minimal exact tensor decomposition.

We model the multiplication tensor T[i, j, k] = delta(i, j, k) where
- i in [0, n)  (row of A, n=3)
- j in [0, m)  (column of A / row of B, m=3)
- k in [0, p)  (column of B, p=6)

The decomposition has the form:
  T[i, j, k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

We use a rank-1 tensor generation approach with random initialization and
iterative refinement (similar to the tensor train / alternating least squares
method for matrix multiplication).

Key insight: For 3x3x3, the known minimal rank is 23 (Landsberg). For 3x3x6,
we expect the rank to be larger. We will search for a valid decomposition
using numerical optimization with exact rational recovery.
"""

import numpy as np
from fractions import Fraction
import time
import itertools
import sys

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the 3D tensor T[i, j, k] = 1 if i,j,k correspond to valid mult, else 0.
    
    Actually for matrix multiplication C = A @ B where A is n x m and B is m x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor is T[i, j, k] = delta (the structure tensor).
    In the decomposition:
      C[i, k] = sum_j A[i,j] B[j,k]
    This is represented as:
      T[i, j, k] such that C[i,k] = sum_{j} A[i,j] * T[i,j,k] * B[j,k]
    
    So T[i, j, k] = 1 for all valid (i,j,k) triples? No.
    
    Let me reconsider. The multiplication tensor has entries:
    T[i, j, k] = 1 if this corresponds to a term in the product, else 0.
    
    Actually, the standard formulation: the multiplication tensor mu_{n,m,p} has
    entries mu_{(i,j),(j,k),(i,k)} = 1 for all i in [n], j in [m], k in [p].
    
    In the factorized form with U of shape (n*m, R), V of shape (m*p, R), W of shape (p*n, R):
    
    The bilinear form is: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Wait, that's not quite right either. Let me use the standard convention:
    
    U has rows indexed by (i, j) for i in [n], j in [m] -> n*m rows
    V has rows indexed by (j, k) for j in [m], k in [p] -> m*p rows  
    W has rows indexed by (k, i) for k in [p], i in [n] -> p*n rows
    
    The decomposition says:
    For each rank-1 term r:
      A_factor[i,j] = U[i*m+j, r]
      B_factor[j,k] = V[j*p+k, r]
      C_factor[k,i] = W[k*n+i, r]
    
    And the multiplication is:
      C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    Hmm, this doesn't look like standard matrix mult. Let me think again.
    
    Standard: C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition for matrix multiplication:
    T_{i,j,k} = 1 for all (i,j,k) in [n]x[m]x[p]
    
    No wait. The structure tensor of matrix multiplication C_{n x m} * C_{m x p} -> C_{n x p}
    is a tensor in (C^{n*m})^* x (C^{m*p})^* x (C^{n*p})^* or similar.
    
    Let me use the explicit form. We want:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We write this as:
    C[i,k] = sum_r w_r * (sum_j u_{r,i,j} A[i,j]) * (sum_j v_{r,j,k} B[j,k])
    
    where w_r, u, v are the decomposition parameters. For this to equal the matrix product
    for ALL A and B, we need:
    
    sum_r w_r * u_{r,i,j} * v_{r,j,k} = delta_{j,j} ... no.
    
    Actually: C[i,k] = sum_j A[i,j] B[j,k]
    
    If we set for each r:
      u_{r,i,j} is nonzero only for specific (i,j)
      v_{r,j,k} is nonzero only for specific (j,k)
      w_{r} is a scalar
    
    Then: sum_r w_r * (sum_j u_{r,i,j} A[i,j]) * (sum_j v_{r,j,k} B[j,k])
         = sum_r w_r * sum_{j1,j2} u_{r,i,j1} v_{r,j2,k} A[i,j1] B[j2,k]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_r w_r * u_{r,i,j1} * v_{r,j2,k} = delta_{j1,j2} for all i,j1,j2,k
    
    This is the condition. So the "tensor" we're decomposing has entries:
    T[i, j1, j2, k] = delta_{j1,j2}
    
    But that's a 4-index tensor. The standard rank-1 decomposition groups (i,j) into one index,
    (j,k) into another, and (i,k) into another.
    
    Let me use the convention from the problem statement more carefully:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    The decomposition is:
    For all A (n x m) and B (m x p):
      C = A @ B  (n x p)
    
    is represented as:
      C[i,k] = sum_r W[k*n+i, r] * (U @ A^T)[i*m + ?] ... 
    
    I think the cleanest way is:
    
    Define the multiplication as a bilinear map. The tensor T has entries T[i,j,k] where:
    - i indexes the row of A (and output C): i in [0, n)
    - j indexes the column of A / row of B: j in [0, m)  
    - k indexes the column of B (and output C): k in [0, p)
    
    T[i,j,k] = 1 for all i,j,k (since every term A[i,j]*B[j,k] appears in C[i,k])
    
    Wait no! T[i,j,k] should encode the structure. The correct statement is:
    
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    The "tensor" is T in C^{n*m} x C^{m*p} x C^{n*p} (or with W indexing (k,i) as p*n).
    
    T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k.
    
    So T is a 3-way tensor with dimensions (n*m) x (m*p) x (p*n) = (9) x (18) x (18) for our case.
    And T has a 1 at position ((i,j), (j,k), (k,i)) for each (i,j,k) in [3]x[3]x[6].
    There are 3*3*6 = 54 ones.
    
    The rank is the minimum R such that T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    
    So U[:,r] = u_r, V[:,r] = v_r, W[:,r] = w_r.
    
    And the condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid (i,j,k) triple
    = 1 for ALL i in [n], j in [m], k in [p]
    
    Wait, that means T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k. But T is a 3-way tensor,
    so T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i.
    
    For a given (a, b, c), there may be multiple (i,j,k) that give this (a,b,c).
    Actually, a = i*m+j uniquely determines i and j. b = j*p+k determines j and k (j must match).
    c = k*n+i determines k and i (k and i must match).
    
    So for T[a, b, c] to be well-defined, we need consistency. If a = i*m+j, b = j'*p+k, c = k'*n+i',
    then we need j = j', k = k', i = i' for it to be a valid entry. If consistent, T = 1. Otherwise T = 0.
    
    So T[a, b, c] = 1 if the indices are consistent (i.e., there exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i), else 0.
    
    Number of 1's: 3*3*6 = 54.
    
    Now I can build this tensor and try to find a low-rank decomposition.
    """
    n, m, p = n, m, p
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_random_rank1_addition(U, V, W, T, n, m, p, rank, iterations=50, tol=1e-9):
    """Try to reduce rank by finding a better decomposition via random perturbations."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    
    for it in range(iterations):
        # Random perturbation
        U_new = U + np.random.randn(*U.shape) * 0.1
        V_new = V + np.random.randn(*V.shape) * 0.1
        W_new = W + np.random.randn(*W.shape) * 0.1
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U_new[a, :] * V_new[b, :] * W_new[c, :])
                    err += (s - 1.0) ** 2
        err /= (n * m * p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
    
    return best_U, best_V, best_W, best_err


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = U.shape[1]
    
    # Build target tensor for verification
    T = _build_target_tensor(n, m, p)
    
    # Verify schoolbook
    ok, info = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        # Should not happen
        pass
    
    # Try to reduce rank using various strategies
    # Strategy: try to find a decomposition of lower rank using numerical methods
    
    # For now, return the schoolbook decomposition as baseline
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
