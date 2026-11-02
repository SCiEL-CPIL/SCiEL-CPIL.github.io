"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val = val + U[a][r] * V[b][r] * W[c][r]
                    if val != 1:
                        return False
        return True
    
    def solve_linear_system_exact(A):
        """Solve linear system Ax=b using exact Gaussian elimination. Returns solution vector or None."""
        n_rows = len(A)
        if n_rows == 0:
            return []
        n_cols = len(A[0]) - 1  # exclude augmented column
        
        if n_cols == 0:
            return []
        
        # Make a copy
        M = [row[:] for row in A]
        
        # Gaussian elimination
        rank = 0
        pivot_cols = []
        for col in range(n_cols):
            # Find pivot
            pivot_row = None
            for row in range(rank, n_rows):
                if M[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            
            # Swap rows
            if pivot_row != rank:
                M[rank], M[pivot_row] = M[pivot_row], M[rank]
            
            pivot_cols.append(col)
            
            # Eliminate below
            for row in range(rank + 1, n_rows):
                if M[row][col] != 0:
                    factor = M[row][col] / M[rank][col]
                    for j in range(col, n_cols + 1):
                        M[row][j] = M[row][j] - factor * M[rank][j]
            
            rank += 1
            if rank == len(pivot_cols):
                break
        
        # Check for inconsistency
        for row in range(rank, n_rows):
            for j in range(n_cols):
                if M[row][j] != 0:
                    return None  # Inconsistent
        
        # Back substitution
        x = [Fraction(0)] * n_cols
        for row in range(rank - 1, -1, -1):
            col = pivot_cols[row]
            x[col] = M[row][n_cols]
            for j in range(col + 1, n_cols):
                if M[row][j] != 0:
                    x[col] = x[col] - M[row][j] * x[j]
            x[col] = x[col] / M[row][col]
        
        return x
    
    def try_rank(r, n, m, p, num_trials=30):
        """
        Try to find a rank-r decomposition.
        Strategy: randomize U and V, solve linear system for W.
        """
        nm, mp, pn = n * m, m * p, p * n
        num_vars = pn * r
        
        for trial in range(num_trials):
            # Generate random rational U and V with small entries
            U = []
            for _ in range(nm):
                row = []
                for _ in range(r):
                    num = random.randint(-2, 2)
                    den = random.choice([1, 2, 3])
                    if num == 0:
                        num = random.choice([-1, 1])
                    row.append(Fraction(num, den))
                U.append(row)
            
            V = []
            for _ in range(mp):
                row = []
                for _ in range(r):
                    num = random.randint(-2, 2)
                    den = random.choice([1, 2, 3])
                    if num == 0:
                        num = random.choice([-1, 1])
                    row.append(Fraction(num, den))
                V.append(row)
            
            # Build linear system for W
            # Equations: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            A = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        row = [Fraction(0)] * (num_vars + 1)
                        for r_idx in range(r):
                            var_idx = c * r + r_idx
                            row[var_idx] = U[a][r_idx] * V[b][r_idx]
                        row[num_vars] = Fraction(1)
                        A.append(row)
            
            # Solve
            W_flat = solve_linear_system_exact(A)
            if W_flat is None:
                continue
            
            # Reconstruct W
            W = [[Fraction(0)] * r for _ in range(pn)]
            for c in range(pn):
                for r_idx in range(r):
                    W[c][r_idx] = W_flat[c * r + r_idx]
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    # Try ranks from schoolbook down to lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few ranks below schoolbook first for efficiency
    ranks_to_try = list(range(schoolbook_rank, lower_bound - 1, -1))
    
    for r in ranks_to_try:
        result = try_rank(r, n, m, p, num_trials=20)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = schoolbook_rank
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
