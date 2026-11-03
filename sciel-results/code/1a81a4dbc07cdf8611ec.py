import time
import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce
import random

_START_TIME = time.time()
DEADLINE = _START_TIME + 118

def _time_left():
    return DEADLINE - time.time()

def construct():
    n, m, p = 3, 3, 4
    
    # Schoolbook fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 12
    for r in range(12, best_rank):
        if _time_left() < 5:
            break
            
        result = _try_rank_search(n, m, p, r)
        if result is not None:
            best_result = result
            best_rank = r
    
    return best_result

def _schoolbook(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _try_rank_search(n, m, p, r):
    np.random.seed(int(time.time() * 1000) % 1000000)
    
    max_attempts = 2000
    if _time_left() < 10:
        max_attempts = 200
        
    for attempt in range(max_attempts):
        if _time_left() < 2:
            break
        
        # Generate random sparse U, V with entries in {-1, 0, 1}
        # Density should be high enough to make the system solvable
        density = 0.5
        
        U_vals = np.zeros((n*m, r), dtype=int)
        V_vals = np.zeros((m*p, r), dtype=int)
        
        for i in range(n*m):
            for c in range(r):
                if np.random.random() < density:
                    U_vals[i, c] = np.random.choice([-1, 1])
                    
        for i in range(m*p):
            for c in range(r):
                if np.random.random() < density:
                    V_vals[i, c] = np.random.choice([-1, 1])
                    
        # Construct the linear system for W
        # Variables: W_{k,i}^{(c)} for c in 0..r-1, k in 0..p-1, i in 0..n-1
        # Total variables: r * p * n
        num_vars = r * p * n
        num_eqs = n * m * p
        
        # A matrix: num_eqs x num_vars
        # b vector: num_eqs
        A = np.zeros((num_eqs, num_vars), dtype=object)
        b = np.zeros(num_eqs, dtype=object)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[eq_idx] = 1
                    for c in range(r):
                        # Coefficient of W_{k,i}^{(c)} is U_{i,j}^{(c)} * V_{j,k}^{(c)}
                        coeff = U_vals[i*m+j, c] * V_vals[j*p+k, c]
                        if coeff != 0:
                            # Variable index for W_{k,i}^{(c)}
                            # Let's index W as W[c, k, i] -> flat index c*p*n + k*n + i
                            var_idx = c * p * n + k * n + i
                            A[eq_idx, var_idx] = coeff
                    eq_idx += 1
                    
        # Solve A * w = b
        # Use Gaussian elimination over Fractions
        try:
            w_solution = _solve_linear_system(A, b)
        except:
            continue
            
        if w_solution is None:
            continue
            
        # Check if w_solution entries are half-integers
        valid = True
        W_vals = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
        for c in range(r):
            for k in range(p):
                for i in range(n):
                    var_idx = c * p * n + k * n + i
                    val = w_solution[var_idx]
                    # Check if val is a half-integer
                    # val = p/q. We want q to be 1 or 2 (after reduction)
                    # Actually, Fraction reduces automatically.
                    if val.denominator not in [1, 2]:
                        valid = False
                        break
                    W_vals[k*n+i][c] = val
                if not valid: break
            if not valid: break
            
        if not valid:
            continue
            
        # Convert to lists
        U_list = [[int(U_vals[i, c]) for c in range(r)] for i in range(n*m)]
        V_list = [[int(V_vals[i, c]) for c in range(r)] for i in range(m*p)]
        W_list = W_vals
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def _solve_linear_system(A, b):
    # Gaussian elimination over Fractions
    # A is list of lists of Fractions/ints
    # b is list of Fractions/ints
    n_rows = len(b)
    n_cols = len(A[0])
    
    # Convert to Fraction
    A_frac = [[Fraction(x) for x in row] for row in A]
    b_frac = [Fraction(x) for x in b]
    
    # Augmented matrix
    M = [row[:] + [b_frac[i]] for i, row in enumerate(A_frac)]
    
    # Forward elimination
    for col in range(n_cols):
        # Find pivot
        pivot_row = -1
        for row in range(col, n_rows):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue # Free variable, we can set it to 0
            
        # Swap rows
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Normalize pivot row
        pivot_val = M[col][col]
        for j in range(col, n_cols + 1):
            M[col][j] /= pivot_val
            
        # Eliminate other rows
        for row in range(n_rows):
            if row != col and M[row][col] != 0:
                factor = M[row][col]
                for j in range(col, n_cols + 1):
                    M[row][j] -= factor * M[col][j]
                    
    # Extract solution
    # Since the system is underdetermined (usually), we set free variables to 0
    solution = [Fraction(0) for _ in range(n_cols)]
    for col in range(n_cols):
        # Find if col is a pivot column
        pivot_row = -1
        for row in range(n_rows):
            if M[row][col] == 1:
                # Check if it's the only non-zero in this column in the reduced form
                # Actually, in RREF, pivot columns have a 1 and 0s elsewhere.
                # We can just take M[row][n_cols] if M[row][col] is 1 and others 0.
                # Simpler: if we have RREF, solution[col] = M[pivot_row][n_cols]
                pass
        # Let's do a simpler extraction for RREF
        # Find row where M[row][col] is 1 and M[r][col] is 0 for all r != row
        is_pivot = True
        pivot_r = -1
        for r in range(n_rows):
            if M[r][col] == 1:
                if pivot_r != -1:
                    is_pivot = False
                    break
                pivot_r = r
            elif M[r][col] != 0:
                is_pivot = False
                break
        if is_pivot and pivot_r != -1:
            solution[col] = M[pivot_r][n_cols]
            
    return solution

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
