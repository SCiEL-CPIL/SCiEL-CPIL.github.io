import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def verify(U, V, W, r):
        """Verify decomposition using exact Fraction arithmetic."""
        T_comp = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        T_comp = [[[Fraction(0) for _ in range(p)] for _ in range(m)] for _ in range(n)]
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    u_val = U[i * m + j][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j * p + k][c]
                        w_val = W[k * n + i][c]
                        T_comp[i][j][k] += u_val * v_val * w_val
        # Check if all entries are 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_comp[i][j][k] != Fraction(1):
                        return False
        return True

    def schoolbook_decomposition():
        """Return the schoolbook rank-45 decomposition."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W, r

    def try_rank_r(r, num_attempts=20):
        """Try to find a rank-r decomposition using structured search."""
        # Strategy: Start with schoolbook and try to merge terms
        # For rank 44, we need to save 1 term from the 45 schoolbook terms.
        # 
        # Idea: Use a "Strassen-like" approach where we introduce a shared variable.
        # 
        # Let's try a different approach: parameterize U and V with small integer/rational
        # entries and solve for W exactly.
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly choose a pattern: most entries are 0 or 1, some are -1 or 2
            # to allow for Strassen-like cancellations
            U = [[Fraction(0) for _ in range(r)] for _ in range(n * m)]
            V = [[Fraction(0) for _ in range(r)] for _ in range(m * p)]
            
            # Initialize with schoolbook-like pattern but with some modifications
            # For each column c, assign it to a (i,j,k) triple, but allow sharing
            col_assignments = []
            for c in range(r):
                # Randomly assign to a (i,j,k) with some probability of sharing
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                col_assignments.append((i, j, k))
            
            # Set U and V based on assignments
            for c, (i, j, k) in enumerate(col_assignments):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
            
            # Now solve for W exactly
            # For each (i,k), we have m equations:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for j=0..m-1
            # This is a linear system in W[k*n+i, 0..r-1]
            
            W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the m x r matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = [[Fraction(0) for _ in range(r)] for _ in range(m)]
                    for j in range(m):
                        for c in range(r):
                            A[j][c] = U[i * m + j][c] * V[j * p + k][c]
                    
                    # Solve A * w = 1_m where w = W[k*n+i, :]
                    # Use exact Gaussian elimination
                    # Augmented matrix: A | 1
                    aug = [row[:] + [Fraction(1)] for row in A]
                    
                    # Gaussian elimination
                    rows = m
                    cols = r
                    pivot_row = 0
                    rank = 0
                    for col in range(cols):
                        # Find pivot
                        pivot = -1
                        for row in range(pivot_row, rows):
                            if aug[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        # Swap
                        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
                        # Eliminate below
                        for row in range(pivot_row + 1, rows):
                            if aug[row][col] != 0:
                                factor = aug[row][col] / aug[pivot_row][col]
                                for cc in range(col, cols + 1):
                                    aug[row][cc] -= factor * aug[pivot_row][cc]
                        pivot_row += 1
                        rank += 1
                    
                    # Check consistency: any row that is all zeros in A part must have 0 in RHS
                    consistent = True
                    for row in range(pivot_row, rows):
                        all_zero = True
                        for col in range(cols):
                            if aug[row][col] != 0:
                                all_zero = False
                                break
                        if all_zero and aug[row][cols] != 0:
                            consistent = False
                            break
                    
                    if not consistent:
                        valid = False
                        break
                    
                    # Back substitution to find one solution
                    w_sol = [Fraction(0) for _ in range(cols)]
                    for row in range(pivot_row - 1, -1, -1):
                        val = aug[row][cols]
                        for cc in range(row + 1, cols):
                            val -= aug[row][cc] * w_sol[cc]
                        if aug[row][row] != 0:
                            w_sol[row] = val / aug[row][row]
                        else:
                            # Free variable, set to 0
                            w_sol[row] = Fraction(0)
                    
                    W[k * n + i] = w_sol
            
            if valid:
                if verify(U, V, W, r):
                    return U, V, W
        
        return None

    # Try ranks from 44 down to 35
    for r in range(44, 34, -1):
        if time.time() > deadline:
            break
        result = try_rank_r(r, num_attempts=10)
        if result is not None:
            U, V, W = result
            return {
                "U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]
            }

    # Fallback to schoolbook
    U, V, W, r = schoolbook_decomposition()
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
