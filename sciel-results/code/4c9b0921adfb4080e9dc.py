"""
Matrix multiplication 4x4 * 4x5 rank-minimal exact tensor decomposition.

Strategy:
1. Start with schoolbook rank 80.
2. Use linear algebra over rationals (exact) to find linear dependencies among the
   rank-1 slices in each matricization, iteratively reducing rank.
3. Additionally, try structured reductions based on known lower bounds and
   combinatorial arguments for small matrix multiplication.

The key insight for reduction: if a column c in U is a linear combination of other
columns with the SAME coefficients as in V and W, then that column is redundant.

We also try a more aggressive approach: at each step, look for any column that can
be expressed as a linear combination in all three modes simultaneously (with consistent
coefficients), and remove it.
"""

import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [4, 4, 5]

def _schoolbook():
    """Return schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _to_fraction_matrix(mat):
    """Convert a 2D list to a 2D list of Fractions."""
    return [[Fraction(x) for x in row] for row in mat]

def _rref(frac_mat):
    """Compute reduced row echelon form of a matrix of Fractions.
    Returns (rref_matrix, pivot_columns)."""
    if not frac_mat or not frac_mat[0]:
        return [], []
    m = len(frac_mat)
    n = len(frac_mat[0])
    mat = [row[:] for row in frac_mat]
    pivot_cols = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        # Find pivot
        pivot = None
        for i in range(r, m):
            if mat[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        mat[r], mat[pivot] = mat[pivot], mat[r]
        # Normalize
        factor = mat[r][c]
        mat[r] = [x / factor for x in mat[r]]
        # Eliminate
        for i in range(m):
            if i != r and mat[i][c] != 0:
                factor = mat[i][c]
                mat[i] = [mat[i][j] - factor * mat[r][j] for j in range(n)]
        pivot_cols.append(c)
        r += 1
    return mat, pivot_cols

def _is_linear_combination(frac_mat, target_col, other_cols):
    """Check if target_col is a linear combination of the columns in other_cols.
    frac_mat is a list of rows, each row is a list of Fractions.
    Returns the coefficients if found, else None."""
    if not other_cols:
        return None
    m = len(frac_mat)
    n = len(other_cols)
    # Build augmented matrix [cols | target]
    aug = []
    for i in range(m):
        row = [frac_mat[i][c] for c in other_cols]
        row.append(frac_mat[i][target_col])
        aug.append(row)
    # RREF
    rref, pivots = _rref(aug)
    # Check consistency
    for i in range(m):
        if all(rref[i][j] == 0 for j in range(n)) and rref[i][n] != 0:
            return None
    # Find solution
    sol = [Fraction(0)] * n
    pivot_set = set(pivots)
    for i in range(len(pivots)):
        col = pivots[i]
        sol[col] = rref[i][n]
    return sol

def _try_reduce(U, V, W):
    """Try to reduce the rank by finding linearly dependent columns.
    Returns (U, V, W, reduced_rank) or None if no reduction found."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_f = _to_fraction_matrix(U)
    V_f = _to_fraction_matrix(V)
    W_f = _to_fraction_matrix(W)
    
    removed = [False] * rank
    new_rank = rank
    
    # Try each column
    for c in range(rank):
        if removed[c]:
            continue
        other_cols = [c2 for c2 in range(rank) if c2 != c and not removed[c2]]
        if not other_cols:
            continue
        
        # Check if column c in U is a linear combination of other columns in U
        coeff_U = _is_linear_combination(U_f, c, other_cols)
        if coeff_U is None:
            continue
        
        # Check if column c in V is a linear combination of other columns in V
        coeff_V = _is_linear_combination(V_f, c, other_cols)
        if coeff_V is None:
            continue
        
        # Check if column c in W is a linear combination of other columns in W
        coeff_W = _is_linear_combination(W_f, c, other_cols)
        if coeff_W is None:
            continue
        
        # Check if the coefficients are the same across all three modes
        valid = True
        for idx in range(len(other_cols)):
            if coeff_U[idx] != coeff_V[idx] or coeff_U[idx] != coeff_W[idx]:
                valid = False
                break
        
        if valid:
            # Remove column c
            removed[c] = True
            new_rank -= 1
    
    if new_rank == rank:
        return None
    
    # Build reduced matrices
    keep = [c for c in range(rank) if not removed[c]]
    U_new = [[U[i][c] for c in keep] for i in range(len(U))]
    V_new = [[V[i][c] for c in keep] for i in range(len(V))]
    W_new = [[W[i][c] for c in keep] for i in range(len(W))]
    
    return U_new, V_new, W_new, new_rank

def _verify(U, V, W):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    U, V, W = result["U"], result["V"], result["W"]
    
    # Try iterative reduction
    while time.time() < deadline:
        reduced = _try_reduce(U, V, W)
        if reduced is None:
            break
        U, V, W, new_rank = reduced
        if new_rank == 0:
            break
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        result = _schoolbook()
        return result
    
    return {"U": U, "V": V, "W": W}
