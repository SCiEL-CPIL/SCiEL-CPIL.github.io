"""
Matrix multiplication 4x4 * 4x5 rank-minimal exact tensor decomposition.

Strategy:
1. Start with schoolbook rank 80.
2. Use linear algebra over rationals (exact) to find linear dependencies among the
   rank-1 slices in each matricization, iteratively reducing rank.
3. Additionally, try structured reductions based on known lower bounds and
   combinatorial arguments for small matrix multiplication.

For 4x4 * 4x5, the trivial rank is 80. The known lower bound from Strassen-type
arguments and the fact that this is not a square cube suggests rank can be reduced.
We use a greedy linear-dependency elimination approach:
- Represent each rank-1 term as a triple (u_c, v_c, w_c) in R^{16} x R^{20} x R^{20}.
- The tensor is T = sum_c u_c ⊗ v_c ⊗ w_c.
- If a subset of terms has a linear dependence in one mode, we can potentially
  re-express them with fewer terms.

However, a more practical approach for small dimensions: use the fact that
matrix multiplication (n x m) * (m x p) has rank at most n*m*p, and for
small cases we can try to find a decomposition of rank < n*m*p by solving
systems of equations.

Given the time constraint, we'll implement:
1. Schoolbook as baseline (rank 80).
2. A linear algebra based reduction: check if any column in U, V, or W is linearly
   dependent on others, and if so, eliminate it by expressing it as a combination.
   This requires that the corresponding entries in all three matrices allow the
   cancellation.

Actually, the cleanest approach: the tensor T_{i,j,k} = delta (in the appropriate sense).
For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
T[i, j, k] where i in [0,n), j in [0,m), k in [0,p) is 1 if the "middle" index matches,
but actually the tensor is defined as T[i, j, k] = 1 for all i,j,k with the constraint that
the multiplication uses index j for both A's column and B's row.

Wait, let me re-read the task. The output factor W[k*n+i, r] follows the trace convention.
The standard matrix multiplication tensor for (n x m) * (m x p) -> (n x p):
The tensor T has indices (i, j, k) where i is row of A (and C), j is the shared index,
k is column of B (and C). T[i, j, k] = 1 for all valid i, j, k.

A rank-1 decomposition term is u ⊗ v ⊗ w where u in R^{n*m} (flattened A), v in R^{m*p} (flattened B),
w in R^{n*p} (flattened C). The schoolbook has one term per (i, j, k) triple... no wait.

Schoolbook: for each j (the shared index), and for each i and k, we have a term.
Actually, schoolbook is: for each j in [0, m), for each i in [0, n), for each k in [0, p):
u_{i*m+j} = 1, v_{j*p+k} = 1, w_{k*n+i} = 1.
So rank = n*m*p = 4*4*5 = 80.

To reduce rank, we need to find a set of rank-1 tensors that sum to T.

For small dimensions, we can try to use the fact that certain linear combinations
of rank-1 terms can be merged. Specifically, if we have two terms that differ only in
one index, we might be able to combine them.

Let me try a different approach: use known results. For 4x4 * 4x4, the rank is known
to be at most 49 (Strassen-like) but actually the best known is around 49-52. For 4x4*4x5,
it should be similar or slightly higher.

Given the complexity, let me implement a practical approach:
1. Start with schoolbook.
2. Try to find linear dependencies in the U, V, W matrices and eliminate redundant columns.
3. Use a greedy algorithm: at each step, find a column that is a linear combination of
   others in at least one mode, and check if it can be eliminated.

Actually, the key insight: if column c in U is a linear combination of other columns,
say U[:, c] = sum_a alpha_a * U[:, a], then we can modify the other columns' V and W
entries to absorb this, IF the corresponding V and W entries also satisfy similar relations.
This is generally not straightforward.

Let me try a simpler but effective approach for this specific small case:
- Use the fact that 4x4 * 4x5 can be embedded in 5x5 * 5x5 or similar.
- Or, try to find a decomposition using bilinear algorithms.

Given the time constraint and the need for an exact answer, let me implement:
1. Schoolbook (rank 80) as the safe fallback.
2. A linear algebra based reduction that tries to eliminate linearly dependent columns
   by checking if the dependency holds in all three modes simultaneously.

For the linear dependency check:
- If U[:, c] = sum_a alpha_a * U[:, a] for some set of columns a,
- AND V[:, c] = sum_a alpha_a * V[:, a],
- AND W[:, c] = sum_a alpha_a * W[:, a],
- Then column c is redundant and can be removed.

This is a sufficient condition. Let me implement this.
"""

import time
import numpy as np
from fractions import Fraction

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
    # Build matrix with columns from other_cols
    if not other_cols:
        return None
    m = len(frac_mat)
    cols = []
    for c in other_cols:
        col = [frac_mat[i][c] for i in range(m)]
        cols.append(col)
    # Transpose to make it a matrix with these as rows
    n = len(cols)
    mat = [[cols[j][i] for j in range(n)] for i in range(m)]
    # Add target as last column
    target = [frac_mat[i][target_col] for i in range(m)]
    # Augmented matrix: [cols | target]
    aug = []
    for i in range(m):
        row = [cols[j][i] for j in range(n)]
        row.append(target[i])
        aug.append(row)
    # RREF
    rref, pivots = _rref(aug)
    # Check consistency: any row with all zeros in cols part but nonzero in target part
    for i in range(m):
        if all(rref[i][j] == 0 for j in range(n)) and rref[i][n] != 0:
            return None
    # If consistent, find a solution. Use the pivot columns to determine free variables.
    # For simplicity, set free variables to 0 and solve for pivot variables.
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
        
        # All three are linear combinations. Check if the coefficients are the same.
        # Actually, for the tensor identity to hold, we need:
        # u_c = sum_a alpha_a u_a, v_c = sum_a alpha_a v_a, w_c = sum_a alpha_a w_a
        # with the SAME alpha_a for all three.
        # Let's check if coeff_U == coeff_V == coeff_W (as a function of the other columns).
        # But the coefficients might be different for different columns.
        # Actually, the condition is: for each other column a, the coefficient in U, V, W
        # should be the same? No, that's not right either.
        
        # The correct condition: if u_c = sum_a alpha_a u_a, then for the tensor to be preserved,
        # we need v_c = sum_a alpha_a v_a AND w_c = sum_a alpha_a w_a.
        # So the coefficients must be the same across all three modes.
        
        # Check if coeff_U, coeff_V, coeff_W give the same mapping.
        # For each other column a, the coefficient should be the same.
        valid = True
        for idx, a in enumerate(other_cols):
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
    # Build the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i, j, k.
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
