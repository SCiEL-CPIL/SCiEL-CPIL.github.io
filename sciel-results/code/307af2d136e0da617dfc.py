"""Find a low-rank exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

def _validate(U, V, W, n, m, p, rank):
    """Exact check: for all i,k,j,l: sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] == delta_{j,l}"""
    for i in range(n):
        for k in range(p):
            w_row = W[k * n + i]
            for j in range(m):
                for l in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[l * p + k][c] * w_row[c]
                    expected = 1 if j == l else 0
                    if s != expected:
                        return False
    return True

def _is_half_integer(val):
    """Check if val is an integer or half-integer (multiple of 0.5)."""
    # val is a Fraction
    num, den = val.numerator, val.denominator
    return den == 1 or den == 2

def _solve_w_system(M, b, rank):
    """Solve M @ w = b where M is (m*p) x rank, b is (m*p).
    Returns a solution w (list of Fractions) or None if inconsistent.
    Uses exact Gaussian elimination.
    """
    num_rows = len(M)
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(M)]
    
    # Gaussian elimination
    pivot_row = 0
    pivot_cols = []
    for col in range(rank):
        # Find pivot
        found = False
        for row in range(pivot_row, num_rows):
            if aug[row][col] != 0:
                found = True
                break
        if not found:
            continue
        # Swap
        if row != pivot_row:
            aug[pivot_row], aug[row] = aug[row], aug[pivot_row]
        # Normalize pivot row
        pv = aug[pivot_row][col]
        for c in range(col, rank + 1):
            aug[pivot_row][c] /= pv
        # Eliminate
        for r in range(num_rows):
            if r != pivot_row and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, rank + 1):
                    aug[r][c] -= factor * aug[pivot_row][c]
        pivot_cols.append(col)
        pivot_row += 1
    
    # Check consistency: rows that are all zeros in coefficient part must have 0 rhs
    for row in range(pivot_row, num_rows):
        if any(aug[row][c] != 0 for c in range(rank)):
            return None
        if aug[row][rank] != 0:
            return None
    
    # Solution: free variables set to 0, pivot variables from augmented matrix
    w = [Fraction(0)] * rank
    for i, col in enumerate(pivot_cols):
        w[col] = aug[i][rank]
    return w

def _try_rank(rank, n, m, p, deadline, rng, max_trials):
    """Try to find a valid rank-r decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Generate structured U and V with small integers
        # Use sparse patterns to increase chance of solvability
        U = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(mp)]
        
        # Ensure no row is all zeros
        for row in range(nm):
            if all(U[row][c] == 0 for c in range(rank)):
                U[row][rng.randint(0, rank-1)] = 1
        for row in range(mp):
            if all(V[row][c] == 0 for c in range(rank)):
                V[row][rng.randint(0, rank-1)] = 1
        
        # Solve for W
        W = [[Fraction(0)] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Build system: for each (j,l): sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] = delta_{j,l}
                M = []
                b = []
                for j in range(m):
                    for l in range(p):
                        row = [Fraction(U[i * m + j][c] * V[l * p + k][c]) for c in range(rank)]
                        M.append(row)
                        b.append(Fraction(1 if j == l else 0))
                
                w_sol = _solve_w_system(M, b, rank)
                if w_sol is None:
                    ok = False
                    break
                W[k * n + i] = w_sol
            if not ok:
                break
        
        if not ok:
            continue
        
        # Check if all W entries are half-integers
        all_half = True
        for i in range(n):
            for k in range(p):
                for c in range(rank):
                    if not _is_half_integer(W[k * n + i][c]):
                        all_half = False
                        break
                if not all_half:
                    break
            if not all_half:
                break
        
        if not all_half:
            continue
        
        # Convert to int/half-int float representation
        W_float = [[float(W[i][c]) for c in range(rank)] for i in range(pn)]
        U_float = [[float(U[i][c]) for c in range(rank)] for i in range(nm)]
        V_float = [[float(V[i][c]) for c in range(rank)] for i in range(mp)]
        
        # Final verification with exact fractions
        U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(nm)]
        V_frac = [[Fraction(V[i][c]) for c in range(rank)] for i in range(mp)]
        
        valid = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = Fraction(0)
                        for c in range(rank):
                            s += U_frac[i * m + j][c] * V_frac[l * p + k][c] * W[k * n + i][c]
                        if s != Fraction(1 if j == l else 0):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Clean up: convert to int where possible
            def clean(val):
                f = float(val)
                if abs(f - round(f)) < 1e-9:
                    return int(round(f))
                return f
            
            U_clean = [[clean(v) for v in row] for row in U_float]
            V_clean = [[clean(v) for v in row] for row in V_float]
            W_clean = [[clean(v) for v in row] for row in W_float]
            
            return U_clean, V_clean, W_clean
    
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)
    
    # Try ranks from 12 upward
    for rank in range(12, 37):
        if time.time() > deadline - 5:
            break
        
        # More trials for lower ranks
        max_trials = 200 if rank <= 15 else 100 if rank <= 20 else 50
        
        result = _try_rank(rank, n, m, p, deadline - 10, rng, max_trials)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
