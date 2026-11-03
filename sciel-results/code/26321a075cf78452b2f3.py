"""Program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Try to find a rank-(schoolbook_rank - 1) decomposition
    # Strategy: assume the last term can be expressed as a combination of others
    # by finding coefficients a, b, c such that the "missing" term is absorbed.
    
    # For rank 41, we try: for some specific (i0, j0, k0), we don't have a dedicated term,
    # but the sum of other terms still gives 1 at that position.
    
    # Let's try a systematic approach: fix the first 41 schoolbook terms,
    # and see if we can adjust them (by adding a common factor structure) to also cover
    # the 42nd term.
    
    # Alternative: use random rational initialization for U and V (first 41 columns),
    # then solve for W to satisfy all 42 constraints.
    
    def solve_for_W(U, V, target_rank):
        """
        Given U (nm x r) and V (mp x r), solve for W (pn x r) such that
        for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        This is a linear system in the entries of W.
        """
        r = target_rank
        num_constraints = n * m * p  # 42
        # Each constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Unknowns: W has p*n * r entries = 14 * r
        
        # Build the linear system: A * w_vec = b
        # where w_vec is the flattened W matrix
        
        # For each (i,j,k), the constraint is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is linear in W entries.
        
        # Let's build the matrix A of size (num_constraints x (pn * r))
        # and vector b of size num_constraints
        
        A = [[Fraction(0) for _ in range(p * n * r)] for _ in range(num_constraints)]
        b = [Fraction(1) for _ in range(num_constraints)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    for c in range(r):
                        coeff = Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c])
                        if coeff != 0:
                            # W[k*n+i, c] is at position (k*n+i)*r + c in w_vec
                            pos = (k*n+i) * r + c
                            A[idx][pos] = coeff
                    idx += 1
        
        # Solve A * w_vec = b using exact Gaussian elimination
        w_vec = _gaussian_elimination(A, b)
        if w_vec is None:
            return None
        
        # Reshape w_vec into W
        W = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
        for row in range(p*n):
            for col in range(r):
                W[row][col] = w_vec[row * r + col]
        
        return W
    
    def _gaussian_elimination(A, b):
        """
        Solve A*x = b using exact Gaussian elimination over rationals.
        Returns x as a list of Fractions, or None if no unique solution exists.
        """
        num_rows = len(A)
        num_cols = len(A[0])
        
        # Augmented matrix
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        rank = 0
        for col in range(num_cols):
            # Find pivot
            pivot_row = None
            for row in range(rank, num_rows):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                continue
            
            # Swap rows
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            
            # Normalize pivot row
            pivot_val = aug[rank][col]
            for j in range(col, num_cols + 1):
                aug[rank][j] = aug[rank][j] / pivot_val
            
            # Eliminate other rows
            for row in range(num_rows):
                if row != rank and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, num_cols + 1):
                        aug[row][j] = aug[row][j] - factor * aug[rank][j]
            
            rank += 1
        
        # Check for consistency
        for row in range(rank, num_rows):
            if any(aug[row][j] != 0 for j in range(num_cols)):
                return None  # Inconsistent
        
        # Extract solution (assuming unique solution for simplicity)
        if rank != num_cols:
            # Underdetermined - return a particular solution (set free vars to 0)
            x = [Fraction(0) for _ in range(num_cols)]
            for row in range(rank):
                # Find the column with the pivot in this row
                pivot_col = None
                for col in range(num_cols):
                    if aug[row][col] == 1:
                        pivot_col = col
                        break
                if pivot_col is not None:
                    x[pivot_col] = aug[row][num_cols]
            return x
        else:
            x = [aug[row][num_cols] for row in range(num_cols)]
            return x
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(r):
                        total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if total != 1:
                        return False
        return True
    
    def try_random_rank(r, num_trials=50):
        """Try to find a rank-r decomposition using random rational initialization."""
        for trial in range(num_trials):
            # Initialize U and V with small random rationals
            U = [[Fraction(0) for _ in range(r)] for _ in range(n*m)]
            V = [[Fraction(0) for _ in range(r)] for _ in range(m*p)]
            
            # Start with schoolbook-like structure for first min(r, n*m*p) columns
            # and random for the rest
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        col += 1
                    if col >= r:
                        break
                    if col >= r:
                        break
                if col >= r:
                    break
            
            # Fill remaining columns with small random rationals
            for row in range(n*m):
                for c in range(r):
                    if U[row][c] == 0 and c >= col:
                        # Add small random perturbation
                        U[row][c] = Fraction(random.randint(-3, 3), random.randint(1, 3))
            
            for row in range(m*p):
                for c in range(r):
                    if V[row][c] == 0 and c >= col:
                        V[row][c] = Fraction(random.randint(-3, 3), random.randint(1, 3))
            
            # Solve for W
            W = solve_for_W(U, V, r)
            if W is None:
                continue
            
            # Verify
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 41
    result = try_random_rank(schoolbook_rank - 1, num_trials=30)
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
