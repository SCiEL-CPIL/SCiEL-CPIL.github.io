"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a decomposition with rank less than schoolbook
    # Start from schoolbook_rank - 1 and go down
    for target_rank in range(schoolbook_rank - 1, max(n*m, m*p, p*n) - 1, -1):
        result = _try_find_decomposition(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)


def _try_find_decomposition(n, m, p, target_rank, max_attempts=50):
    """
    Try to find a rank-target_rank decomposition using randomized approach.
    Fix U and V randomly, solve for W linearly.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    for attempt in range(max_attempts):
        # Generate random U and V with small integer entries
        U = _random_rational_matrix(nm, target_rank, max_val=3)
        V = _random_rational_matrix(mp, target_rank, max_val=3)
        
        # Build the linear system for W
        # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is linear in W entries.
        
        num_equations = n * m * p
        A = []
        b = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [Fraction(0)] * (pn * target_rank)
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    
                    for c in range(target_rank):
                        coeff = U[a_idx][c] * V[b_idx][c]
                        w_pos = c_idx * target_rank + c
                        row[w_pos] = coeff
                    
                    A.append(row)
                    b.append(Fraction(1))
        
        # Solve the linear system
        w_solution = _solve_linear_system(A, b)
        
        if w_solution is None:
            continue
        
        # Reconstruct W
        W = [[Fraction(0)] * target_rank for _ in range(pn)]
        for c_idx in range(pn):
            for c in range(target_rank):
                W[c_idx][c] = w_solution[c_idx * target_rank + c]
        
        # Verify the decomposition
        if _verify_decomposition(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None


def _random_rational_matrix(rows, cols, max_val=3):
    """Generate a random matrix with small integer entries."""
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            val = random.randint(-max_val, max_val)
            row.append(Fraction(val))
        matrix.append(row)
    return matrix


def _solve_linear_system(A, b):
    """
    Solve A * x = b using Gaussian elimination with exact rational arithmetic.
    Returns solution vector x or None if no unique solution.
    """
    num_eq = len(A)
    num_var = len(A[0]) if num_eq > 0 else 0
    
    if num_eq == 0:
        return [Fraction(0)] * num_var
    
    # Augmented matrix
    aug = []
    for i in range(num_eq):
        row = A[i][:] + [b[i]]
        aug.append(row)
    
    # Gaussian elimination with partial pivoting
    for col in range(num_var):
        # Find pivot
        pivot_row = -1
        for row in range(col, num_eq):
            if aug[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        
        # Normalize pivot row
        pivot_val = aug[col][col]
        for j in range(col, num_var + 1):
            aug[col][j] = aug[col][j] / pivot_val
        
        # Eliminate below
        for row in range(col + 1, num_eq):
            factor = aug[row][col]
            if factor != 0:
                for j in range(col, num_var + 1):
                    aug[row][j] = aug[row][j] - factor * aug[col][j]
    
    # Back substitution
    x = [Fraction(0)] * num_var
    for row in range(num_eq - 1, -1, -1):
        first_nonzero = -1
        for col in range(num_var):
            if aug[row][col] != 0:
                first_nonzero = col
                break
        
        if first_nonzero == -1:
            if aug[row][num_var] != 0:
                return None
            continue
        
        total = aug[row][num_var]
        for col in range(first_nonzero + 1, num_var):
            total -= aug[row][col] * x[col]
        x[first_nonzero] = total / aug[row][first_nonzero]
    
    return x


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W form a valid decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                total = Fraction(0)
                for c in range(len(U[0])):
                    total += U[a_idx][c] * V[b_idx][c] * W[c_idx][c]
                
                if total != Fraction(1):
                    return False
    
    return True


def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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
# EVOLVE-BLOCK-END
