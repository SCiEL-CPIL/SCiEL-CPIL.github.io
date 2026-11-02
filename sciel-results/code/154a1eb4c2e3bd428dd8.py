import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 4, 5]

def _solve_linear(M, b):
    """
    Solve M * x = b for x using Gaussian elimination with Fractions.
    M is list of lists of Fractions (size p x r).
    b is list of Fractions (size p).
    Returns a solution x (list of Fractions) if exists, else None.
    Free variables are set to 0.
    """
    p = len(b)
    r = len(M[0])
    
    # Augmented matrix
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    
    pivot_rows = []
    pivot_cols = []
    
    row_idx = 0
    for col in range(r):
        if row_idx >= p:
            break
        # Find pivot in column col, from row_idx to p-1
        pivot = -1
        for i in range(row_idx, p):
            if A[i][col] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        
        # Swap
        A[row_idx], A[pivot] = A[pivot], A[row_idx]
        
        # Eliminate below
        for i in range(row_idx + 1, p):
            if A[i][col] != 0:
                factor = A[i][col] / A[row_idx][col]
                for j in range(col, r + 1):
                    A[i][j] -= factor * A[row_idx][j]
        
        pivot_rows.append(row_idx)
        pivot_cols.append(col)
        row_idx += 1
        
    # Check consistency
    for i in range(row_idx, p):
        if A[i][r] != 0:
            return None
            
    # Back substitution
    x = [Fraction(0) for _ in range(r)]
    
    # Iterate pivot rows in reverse
    for idx in range(len(pivot_rows) - 1, -1, -1):
        r_idx = pivot_rows[idx]
        c_idx = pivot_cols[idx]
        
        # Sum of known terms
        s = A[r_idx][r]
        for j in range(c_idx + 1, r):
            if x[j] != 0:
                s -= A[r_idx][j] * x[j]
                
        x[c_idx] = s / A[r_idx][c_idx]
        
    return x

def _check_half_integer(vals):
    for v in vals:
        if v.denominator not in (1, 2):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try ranks from low to high
    # Start from 20
    for r in range(20, 61):
        if time.time() > deadline:
            break
            
        # Try multiple random instances of V, W
        # Each attempt is fast (12 small linear systems)
        # We can do many attempts
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            if attempts > 50000: # Safety cap
                break
                
            # Generate random V, W with half-integer entries
            # Use small range
            vals = [Fraction(-2), Fraction(-3, 2), Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)]
            
            V = [[random.choice(vals) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(p * n)]
            
            U_solutions = []
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Construct M (p x r)
                    M = []
                    for k in range(p):
                        row = []
                        for c in range(r):
                            row.append(V[j * p + k][c] * W[k * n + i][c])
                        M.append(row)
                    
                    b = [Fraction(1) for _ in range(p)]
                    
                    sol = _solve_linear(M, b)
                    if sol is None:
                        valid = False
                        break
                    if not _check_half_integer(sol):
                        valid = False
                        break
                    U_solutions.append(sol)
                        
            if not valid:
                continue
                
            # Construct U
            U = []
            idx = 0
            for i in range(n):
                for j in range(m):
                    U.append(U_solutions[idx])
                    idx += 1
            
            return {
                "U": U,
                "V": V,
                "W": W
            }
            
    # Fallback to schoolbook
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
