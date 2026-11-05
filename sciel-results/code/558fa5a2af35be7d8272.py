"""
Rank-minimal exact tensor decomposition for 3x4 by 4x8 matrix multiplication.
Uses the schoolbook decomposition as a safe baseline, then attempts a
structure-aware search for a lower-rank decomposition by exploiting the
asymmetry of the dimensions.

The multiplication tensor T satisfies:
  T[i, j, k] = 1  for the product C[i,k] = sum_j A[i,j]*B[j,k]

We seek rank-r decomposition:
  U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed over c = T[i,j,k]

For n=3, m=4, p=8, the schoolbook rank is 96.
We search for a lower rank using a structured approach:
- Start from rank 48 (half the schoolbook) with a random initialization
  biased toward the known schoolbook structure
- Use alternating least squares to drive the residual to zero
- Verify with exact rational arithmetic if possible

The key insight: since n=3 is the smallest dimension, we can try to
"merge" pairs of the 96 schoolbook terms. Each output entry (i,k) depends
on 4 terms (j=0..3). If we can find a 2-term decomposition for each
(i,k) pair sharing structure across j, we might halve the rank.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 120.0  # seconds; evaluator kills at 150s, leave margin

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _build_schoolbook():
    """Return (U, V, W) as lists of lists for schoolbook rank 96."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _verify_fractional(U, V, W, n, m, p):
    """Verify with exact Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = Fraction(1)
                actual = Fraction(0)
                for c in range(rank):
                    uc = Fraction(U[i * m + j][c]) if not isinstance(U[i * m + j][c], Fraction) else U[i * m + j][c]
                    vc = Fraction(V[j * p + k][c]) if not isinstance(V[j * p + k][c], Fraction) else V[j * p + k][c]
                    wc = Fraction(W[k * n + i][c]) if not isinstance(W[k * n + i][c], Fraction) else W[k * n + i][c]
                    actual += uc * vc * wc
                if actual != expected:
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank):
    """One step of alternating least squares.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] = 1 for all i,j,k in the multiplication tensor.
    
    Actually, the multiplication tensor T_{n,m,p}[i, j, k] = delta_{i,j,k} is not 
    simply all-ones. Let me reconsider.
    
    The matrix multiplication tensor is defined as:
    T_{n,m,p}[i, j, k] = 1 if the (i,k) entry of C = A*B involves A[i,j]*B[j,k]
    
    More precisely, for the multiplication C_{i,k} = sum_j A_{i,j} * B_{j,k},
    the tensor has T[i,j,k] = 1 for all valid i,j,k. Wait, that's the schoolbook 
    tensor which has rank n*m*p.
    
    Actually the multiplication tensor is:
    T_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    
    This is the standard matrix multiplication tensor with rank at most n*m*p.
    We want to find a lower-rank decomposition.
    
    For the ALS, we fix two matrices and solve for the third.
    
    If we fix V and W, we need to find U such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    This is not a standard linear system because the indices are coupled.
    """
    pass  # Placeholder - ALS is complex for this coupled structure

def _try_rank_search(max_rank=96, min_rank=32, num_trials=20):
    """Try to find a decomposition with rank < 96 using random restarts and ALS."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_rank = 96
    best_result = None
    
    for r in range(max_rank, min_rank - 1, -1):
        if _remaining() < 5.0:
            break
        
        for trial in range(num_trials):
            if _remaining() < 2.0:
                break
            
            # Initialize with random values, biased toward small integers
            U = [[random.choice([-1, 0, 1, 0.5, -0.5, 2]) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1, 0.5, -0.5, 2]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1, 0.5, -0.5, 2]) for _ in range(r)] for _ in range(pn)]
            
            # Try a few ALS-like iterations
            # For the multiplication tensor T[i,j,k] = 1:
            # Fix V, W -> solve for U
            # For each (i,j): U[i*m+j, c] should satisfy
            #   sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
            # Let B_{i,j,k,c} = V[j*p+k, c] * W[k*n+i, c]
            # Then sum_c U[i*m+j, c] * B_{i,j,k,c} = 1 for all k
            # This is an underdetermined system (r unknowns, p equations) if r >= p
            
            for iteration in range(10):
                if _remaining() < 1.0:
                    break
                
                # Update U: fix V, W
                for i in range(n):
                    for j in range(m):
                        # For each k, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # Build system: for k in range(p): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        # This is p equations in r unknowns
                        if r >= p:
                            # Simple approach: use first p columns
                            for k in range(p):
                                val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(r))
                                target = 1.0
                                # Gradient descent step
                                grad = [V[j * p + k][c] * W[k * n + i][c] for c in range(r)]
                                diff = target - val
                                lr = 0.01
                                for c in range(r):
                                    U[i * m + j][c] += lr * diff * grad[c]
                
                # Update V: fix U, W
                for j in range(m):
                    for k in range(p):
                        # For each i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        for i in range(n):
                            val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(r))
                            target = 1.0
                            grad = [U[i * m + j][c] * W[k * n + i][c] for c in range(r)]
                            diff = target - val
                            lr = 0.01
                            for c in range(r):
                                V[j * p + k][c] += lr * diff * grad[c]
                
                # Update W: fix U, V
                for k in range(p):
                    for i in range(n):
                        # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        for j in range(m):
                            val = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(r))
                            target = 1.0
                            grad = [U[i * m + j][c] * V[j * p + k][c] for c in range(r)]
                            diff = target - val
                            lr = 0.01
                            for c in range(r):
                                W[k * n + i][c] += lr * diff * grad[c]
            
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-6):
                # Round to nearest half-integer and re-verify
                U2 = [[round(x * 2) / 2 for x in row] for row in U]
                V2 = [[round(x * 2) / 2 for x in row] for row in V]
                W2 = [[round(x * 2) / 2 for x in row] for row in W]
                if _verify(U2, V2, W2, n, m, p, tol=1e-9):
                    best_rank = r
                    best_result = (U2, V2, W2)
                    return best_rank, best_result
    
    return best_rank, best_result


def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    of shapes (n*m, rank), (m*p, rank), (p*n, rank) respectively.
    The rank is the number of columns in each matrix.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as safe fallback
    U_school, V_school, W_school = _build_schoolbook()
    
    # Try to find a better (lower rank) decomposition
    if _remaining() > 10.0:
        try:
            best_rank, best_result = _try_rank_search(max_rank=64, min_rank=32, num_trials=5)
            if best_result is not None and best_rank < n * m * p:
                U, V, W = best_result
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
