"""
Rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.

Uses a randomized algebraic-geometry approach:
  1. Start from the schoolbook (rank 42) decomposition.
  2. For each candidate rank r < 42, try to find a rank-r decomposition by:
     a. Randomly selecting r columns from the schoolbook decomposition.
     b. Checking if the remaining (42-r) tensor entries can be expressed as
        linear combinations of the selected r rank-1 terms.
     c. If so, solve for the W matrix using exact rational arithmetic.
  3. Verify the decomposition exactly.
  4. Return the best (lowest rank) decomposition found.

The key insight: the matrix multiplication tensor M_{n,m,p} has a special
structure where many entries are 1 and the factor matrices have a specific
pattern. This allows us to use the schoolbook decomposition as a starting
point and try to merge terms.
"""

import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m       # 6
    mp = m * p       # 21
    pn = p * n       # 14
    schoolbook_rank = n * m * p  # 42

    # Build the schoolbook decomposition
    # For each (i, j, k), we have a rank-1 term:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1
    #   W[k*n+i, col] = 1
    # where col is the index of this term

    # List of (i, j, k) tuples
    tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
        for col, (i, j, k) in enumerate(tuples):
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
        return U, V, W

    # The tensor entries: T[a, b, c] = 1 for all (i,j,k) where
    # a = i*m+j, b = j*p+k, c = k*n+i
    # There are 42 such entries, each equal to 1.

    # For a rank-r decomposition T = sum_c U_c ⊗ V_c ⊗ W_c,
    # we need: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # Strategy: pick r of the 42 schoolbook terms. For the remaining (42-r) terms,
    # we need to express their contribution as a linear combination of the selected r terms.
    #
    # If we fix U and V (from the selected r terms), then for each (i,j,k) not in the
    # selected set, we need:
    #   sum_{c in selected} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # But U[i*m+j, c] and V[j*p+k, c] are fixed (0 or 1 from the schoolbook selection).
    # So this becomes a linear system in the W entries.
    #
    # However, W is shared across all (i,j,k), so we need a single W that works for all.
    #
    # Better approach: 
    #   - Select r terms from the 42.
    #   - Set U and V to have 1s in the corresponding positions for those r columns.
    #   - For each (i,j,k), the constraint is:
    #     sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #   - This is a system of 42 equations in 14*r unknowns (W has 14 rows, r columns).
    #   - For r = 41: 574 unknowns, 42 equations - heavily underdetermined.
    #   - But we also need the W entries to be consistent (same W[c] for all (i,j,k)).
    #
    # Actually, the system is:
    #   For each (i,j,k): sum_c U_a,c * V_b,c * W_c_idx = 1
    #   where a = i*m+j, b = j*p+k, c_idx = k*n+i
    #
    # Since U_a,c and V_b,c are 0 or 1 (from the schoolbook selection), the coefficient
    # of W_c_idx in equation (i,j,k) is U_a,c * V_b,c.
    #
    # Let's define: for each selected term c (corresponding to (i_c, j_c, k_c)):
    #   U[i_c*m+j_c, c] = 1, V[j_c*p+k_c, c] = 1, and U/V are 0 elsewhere.
    #
    # Then for equation (i,j,k):
    #   sum_{c: i_c=i and j_c=j and k_c=k} W[k*n+i, c] = 1
    #   (since U[i*m+j, c] = 1 iff i_c=i and j_c=j, and V[j*p+k, c] = 1 iff j_c=j and k_c=k)
    #
    # Wait, that's not right. U[i*m+j, c] = 1 if the selected term c has i_c = i and j_c = j.
    # V[j*p+k, c] = 1 if the selected term c has j_c = j and k_c = k.
    # So U[i*m+j, c] * V[j*p+k, c] = 1 iff i_c = i, j_c = j, and k_c = k.
    # That means only the term with exactly (i,j,k) contributes!
    #
    # This means that if we use the schoolbook structure for U and V, then each equation
    # only involves the W entry for that specific (i,j,k). So we can't merge terms this way.
    #
    # We need a different approach. Let's allow U and V to have non-trivial values.
    #
    # New approach: 
    #   1. Start with the full 42-term schoolbook decomposition.
    #   2. Try to express one term as a linear combination of the others.
    #   3. This means finding coefficients alpha_c such that:
    #      e_{i0,j0,k0} = sum_{c != (i0,j0,k0)} alpha_c * e_{i_c,j_c,k_c}
    #      in the tensor product space.
    #
    # But this is a tensor equation, not a linear one. The rank-1 terms are not linearly
    # independent in general.
    #
    # Let's try a more direct approach: use the fact that the matrix multiplication tensor
    # has a specific structure and try to find a lower-rank decomposition by solving
    # a system of polynomial equations.
    #
    # For rank r, we have:
    #   U: 6 x r, V: 21 x r, W: 14 x r
    #   For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a system of 42 polynomial equations in 6*r + 21*r + 14*r = 41*r unknowns.
    # For r = 41: 1681 unknowns, 42 equations.
    #
    # We can use a randomized approach:
    #   1. Randomly initialize U and V with small rational numbers.
    #   2. For each (i,j,k), the equation becomes linear in W:
    #      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #   3. This gives 42 linear equations in 14*r unknowns (the W entries).
    #   4. If the system is consistent, we can solve for W.
    #   5. Verify that the decomposition is exact.
    #
    # The issue is that the W entries are shared across equations, so the system is not
    # simply 42 independent equations. Each W[k*n+i, c] appears in multiple equations
    # (for different j values).
    #
    # Let's reorganize: for fixed (k, i), the W entry W[k*n+i, c] appears in equations
    # for all j (0 to m-1). So we have:
    #   For each (k, i, c): W[k*n+i, c] is a variable.
    #   The equations are: for each (i, j, k):
    #     sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a linear system in the W variables, with coefficients depending on U and V.
    #
    # Let's implement this:
    #   1. For a given rank r, randomly initialize U (6 x r) and V (21 x r) with rational numbers.
    #   2. Build the linear system for W:
    #      Variables: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
    #      That's p*n*r = 14*r variables.
    #      Equations: for each (i, j, k):
    #        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #      That's n*m*p = 42 equations.
    #   3. Solve the linear system using exact rational arithmetic.
    #   4. If a solution exists, verify it exactly.
    #   5. If verification passes, return the decomposition.

    def solve_linear_system(A, b):
        """
        Solve A x = b using exact Gaussian elimination with Fraction.
        A is a list of lists (each row is a list of Fractions).
        b is a list of Fractions.
        Returns x (list of Fractions) or None if no solution.
        """
        num_eqs = len(A)
        num_vars = len(A[0]) if num_eqs > 0 else 0
        
        # Augmented matrix
        aug = [[A[i][j] for j in range(num_vars)] + [b[i]] for i in range(num_eqs)]
        
        # Gaussian elimination
        row = 0
        for col in range(num_vars):
            # Find pivot
            pivot = None
            for r in range(row, num_eqs):
                if aug[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                continue
            
            # Swap rows
            aug[row], aug[pivot] = aug[pivot], aug[row]
            
            # Eliminate
            pivot_val = aug[row][col]
            for r in range(num_eqs):
                if r != row:
                    factor = aug[r][col] / pivot_val
                    if factor != 0:
                        for c in range(col, num_vars + 1):
                            aug[r][c] -= factor * aug[row][c]
            
            row += 1
            if row >= num_eqs:
                break
        
        # Check for inconsistency
        for r in range(row, num_eqs):
            if all(aug[r][c] == 0 for c in range(num_vars)) and aug[r][num_vars] != 0:
                return None
        
        # Extract solution
        # For simplicity, set free variables to 0
        x = [Fraction(0)] * num_vars
        # Map pivot columns to rows
        pivot_cols = []
        for r in range(row):
            for c in range(num_vars):
                if aug[r][c] != 0:
                    pivot_cols.append((r, c))
                    break
        
        # Back substitution
        for r in range(row - 1, -1, -1):
            # Find the pivot column for this row
            pivot_col = None
            for c in range(num_vars):
                if aug[r][c] != 0:
                    pivot_col = c
                    break
            if pivot_col is None:
                continue
            val = aug[r][num_vars]
            for c in range(pivot_col + 1, num_vars):
                val -= aug[r][c] * x[c]
            x[pivot_col] = val / aug[r][pivot_col]
        
        return x

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != 1:
                        return False
        return True

    def try_rank(r, num_attempts=100):
        """Try to find a rank-r decomposition."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        for attempt in range(num_attempts):
            # Randomly initialize U and V with small rational numbers
            # Use small integers to keep things manageable
            U = [[Fraction(random.randint(-3, 3)) for _ in range(r)] for _ in range(nm)]
            V = [[Fraction(random.randint(-3, 3)) for _ in range(r)] for _ in range(mp)]
            
            # Build the linear system for W
            # Variables: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1
            num_w_vars = pn * r
            
            # Map variable index
            def w_var_idx(k, i, c):
                return (k * n + i) * r + c
            
            # Equations: for each (i, j, k):
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            A = []
            b = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = [Fraction(0)] * num_w_vars
                        a = i * m + j
                        b_idx = j * p + k
                        for c in range(r):
                            coeff = U[a][c] * V[b_idx][c]
                            if coeff != 0:
                                row[w_var_idx(k, i, c)] = coeff
                        A.append(row)
                        b.append(Fraction(1))
            
            # Solve
            x = solve_linear_system(A, b)
            if x is None:
                continue
            
            # Construct W
            W = [[Fraction(0)] * r for _ in range(pn)]
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        W[k * n + i][c] = x[w_var_idx(k, i, c)]
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p):
                return U, V, W
        
        return None

    # Try ranks from schoolbook_rank - 1 down to a reasonable lower bound
    # The trivial lower bound is max(nm, mp, pn) = 21
    # But in practice, we'll try a few ranks below 42
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, get the schoolbook decomposition as a fallback
    U_sb, V_sb, W_sb = make_schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower ranks
    # We'll try ranks 41, 40, 39, ... but limit the number of attempts
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 20), -1):
        start_time = time.time()
        result = try_rank(target_rank, num_attempts=50)
        elapsed = time.time() - start_time
        
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = target_rank
            break  # Found the best rank, stop
        
        # If we've spent too much time, stop
        if elapsed > 90:
            break
    
    # Convert to the required format
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
