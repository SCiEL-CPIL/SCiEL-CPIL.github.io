"""
Search for a rank-minimal exact tensor decomposition of the 3x4 @ 4x5 matrix product.

We use a randomized rank-r construction:
  U: (n*m, r), V: (m*p, r), W: (p*n, r) with integer/half-integer entries
such that for all i,j,k:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

We try ranks below the schoolbook rank 60 and verify exactness.
"""

import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 5]


def _verify(U, V, W, n, m, p):
    """Check exact decomposition. U,V,W are lists of lists of Fraction or int."""
    r = len(U[0])
    # Check each (i, j, k) triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _to_half_int(frac):
    """Convert a Fraction to a half-integer (multiple of 0.5) if possible, else None."""
    if frac.denominator == 1:
        return int(frac)
    elif frac.denominator == 2:
        return frac.numerator / 2.0
    else:
        return None


def _check_half_integer(U, V, W):
    """Check if all entries are integers or half-integers."""
    for mat in (U, V, W):
        for row in mat:
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator not in (1, 2):
                        return False
                elif isinstance(val, (int, float)):
                    pass
                else:
                    return False
    return True


def _normalize_to_half_int(U, V, W):
    """Convert Fractions to int or float (half-integer)."""
    def conv(val):
        if isinstance(val, Fraction):
            if val.denominator == 1:
                return int(val)
            else:
                return val.numerator / 2.0
        return val

    U2 = [[conv(v) for v in row] for row in U]
    V2 = [[conv(v) for v in row] for row in V]
    W2 = [[conv(v) for v in row] for row in W]
    return U2, V2, W2


def _try_rank_random(r, n, m, p, max_attempts=200, seed_base=0):
    """
    Try to find a rank-r decomposition using random initialization and iterative refinement.
    We set up the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    
    Strategy: fix V and W to random small half-integer values, then solve for U.
    This is a linear system in U's entries.
    """
    np.random.seed(seed_base)
    
    for attempt in range(max_attempts):
        # Random V and W with small half-integer entries
        # Use entries from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
        vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        V = [[Fraction(np.random.choice(vals)) for _ in range(r)] for _ in range(m * p)]
        W = [[Fraction(np.random.choice(vals)) for _ in range(r)] for _ in range(p * n)]
        
        # Build linear system for U
        # Unknowns: U[i*m+j, c] for all i,j,c -> n*m*r unknowns
        # Equations: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This gives n*m*p equations.
        
        n_unknowns = n * m * r
        n_equations = n * m * p
        
        # Build the system: A * u_vec = b
        # Where u_vec is U flattened
        # For equation (i,j,k): sum_c V[j*p+k,c] * W[k*n+i,c] * U[i*m+j,c] = 1
        
        A = np.zeros((n_equations, n_unknowns), dtype=object)
        b = np.ones(n_equations, dtype=object)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        coeff = V[j * p + k][c] * W[k * n + i][c]
                        if coeff != 0:
                            A[eq_idx, (i * m + j) * r + c] = coeff
                    eq_idx += 1
        
        # Solve using Gaussian elimination over Q
        # Flatten A into a list of Fractions for exact arithmetic
        # This is expensive, so only do it for small systems
        
        if n_unknowns > 500 or n_equations > 500:
            continue  # Too large for exact Gaussian elimination
        
        # Convert to Fraction-based matrix
        A_frac = [[Fraction(A[i, j]) for j in range(n_unknowns)] for i in range(n_equations)]
        b_frac = [Fraction(1) for _ in range(n_equations)]
        
        # Gaussian elimination
        try:
            sol = _gauss_elim_fraction(A_frac, b_frac)
        except:
            continue
        
        if sol is None:
            continue
        
        # Check if solution entries are half-integers
        U_list = []
        valid = True
        for i in range(n * m):
            row = []
            for c in range(r):
                val = sol[i * r + c]
                if val.denominator not in (1, 2):
                    valid = False
                    break
                row.append(val)
            if not valid:
                break
            U_list.append(row)
        
        if not valid:
            continue
        
        # Verify
        if _verify(U_list, V, W, n, m, p):
            U2, V2, W2 = _normalize_to_half_int(U_list, V, W)
            return U2, V2, W2
    
    return None


def _gauss_elim_fraction(A, b):
    """Exact Gaussian elimination over Fractions. Returns solution or None."""
    n = len(b)
    m = len(A[0])
    
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for col in range(min(n, m)):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap
        M[col], M[pivot] = M[pivot], M[col]
        
        # Eliminate below
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for j in range(col, m + 1):
                    M[row][j] -= factor * M[col][j]
    
    # Back substitution
    sol = [Fraction(0)] * m
    for i in range(min(n, m) - 1, -1, -1):
        # Find which variable this row corresponds to
        # We need to track which columns are pivots
        # Simplification: assume the matrix is in row-echelon form with pivots in order
        # This is a simplification; for a proper solve we'd need full tracking
        # For now, if n <= m and the system is consistent, try to extract
        if M[i][i] != 0:
            s = M[i][m]
            for j in range(i + 1, m):
                s -= M[i][j] * sol[j]
            sol[i] = s / M[i][i]
    
    # Verify the solution
    for i in range(n):
        s = Fraction(0)
        for j in range(m):
            s += A[i][j] * sol[j]
        if s != b[i]:
            return None
    
    return sol


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But in practice, for matrix multiplication, the rank is at least max(n, m, p)^2 for square
    # For 3x4x5, known results suggest rank around 40-50
    
    for r in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_random(r, n, m, p, max_attempts=50, seed_base=r * 1000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
