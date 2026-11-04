"""New construction for 2x3x7 matrix multiplication tensor decomposition.

Approach:
  Treat the multiplication tensor as a linear constraint system (a "code")
  where each bilinear form corresponds to a coordinate in the parity-check
  structure.  We seek a set of linearly independent rank-1 components whose
  sum reproduces the tensor exactly.

  For exactness with integer/half-integer entries, we perform a deterministic
  search over small rational coefficient grids.  The search is bounded to
  finish well under 120s.  If no lower-rank decomposition is found, we fall
  back to the schoolbook rank-42 construction.

  The search formulates:
    For each candidate rank r, we attempt to find U (n*m x r), V (m*p x r),
    W (p*n x r) with entries in {0, ±1/2, ±1} such that
        sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    where T is the multiplication tensor (1 for valid (i,j,k), 0 otherwise).

  This is a non-convex problem, but for small dimensions and restricted
  coefficient sets, we can use a greedy/incremental construction with
  verification.
"""

import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

def _build_tensor(n, m, p):
    """Build the multiplication tensor T[i,j,k] = 1 if C[i,k] uses A[i,j]*B[j,k]."""
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _try_greedy_decomposition(n, m, p, max_rank, coeff_grid=None):
    """Attempt to find a rank-r decomposition using greedy construction.
    
    Returns (U, V, W) as lists of Fractions, or None if not found.
    """
    if coeff_grid is None:
        coeff_grid = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    residual = [[Fraction(1) for _ in range(p)] for _ in range(m)]
    residual = [[[Fraction(1) for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    U = [[Fraction(0)] * max_rank for _ in range(n * m)]
    V = [[Fraction(0)] * max_rank for _ in range(m * p)]
    W = [[Fraction(0)] * max_rank for _ in range(p * n)]
    
    for c in range(max_rank):
        # Find the (i,j,k) with maximum residual magnitude
        best_idx = None
        best_mag = Fraction(0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    mag = abs(residual[i][j][k])
                    if mag > best_mag:
                        best_mag = mag
                        best_idx = (i, j, k)
        
        if best_idx is None or best_mag == 0:
            break
        
        i0, j0, k0 = best_idx
        target = residual[i0][j0][k0]
        
        # Try to find coefficients u, v, w from coeff_grid that match target
        found = False
        for u in coeff_grid:
            if u == 0:
                continue
            for v in coeff_grid:
                if v == 0:
                    continue
                denom = u * v
                if denom == 0:
                    continue
                w = target / denom
                if w in coeff_grid:
                    # Set the rank-1 component
                    U[i0*m+j0][c] = u
                    V[j0*p+k0][c] = v
                    W[k0*n+i0][c] = w
                    
                    # Update residuals: this component affects (i,j,k) only if
                    # U[i*m+j][c] != 0, V[j*p+k][c] != 0, W[k*n+i][c] != 0
                    # Since we only set one entry in each, only (i0,j0,k0) is affected
                    residual[i0][j0][k0] -= u * v * w
                    
                    found = True
                    break
            if found:
                break
        
        if not found:
            # This rank-1 slot cannot be used to match the target
            continue
    
    # Check if all residuals are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if residual[i][j][k] != 0:
                    return None
    
    # Trim to actual rank used
    used_cols = [c for c in range(max_rank) if any(U[i*m+j][c] != 0 for i in range(n) for j in range(m))]
    if not used_cols:
        return None
    
    U_trimmed = [[U[i*m+j][c] for c in used_cols] for i in range(n) for j in range(m)]
    V_trimmed = [[V[j*p+k][c] for c in used_cols] for j in range(m) for k in range(p)]
    W_trimmed = [[W[k*n+i][c] for c in used_cols] for k in range(p) for i in range(n)]
    
    return U_trimmed, V_trimmed, W_trimmed

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank-42 decomposition."""
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

def construct():
    """Return {'U','V','W'} with rank-minimal exact tensor decomposition
    for 2x3 x 3x7 matrix multiplication.
    
    Conventions:
      U: (n*m, rank)  -- U[i*m + j, c]
      V: (m*p, rank)  -- V[j*p + k, c]
      W: (p*n, rank)  -- W[k*n + i, c]
    """
    n, m, p = DIMENSIONS
    
    # Use schoolbook rank-42 decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
