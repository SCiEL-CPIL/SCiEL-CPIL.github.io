import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Check: T[i,j,k,l] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+l, c] for c in range(rank))
                    expected = 1.0 if i == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook decomposition as fallback
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    if verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist()):
        pass # It's valid

    # Try to find lower rank decompositions
    # Theoretical lower bound is 15
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            # Random U and V with small integer entries
            # Using {0, 1, -1} to keep numbers small
            U_frac = [[Fraction(random.choice([0, 1, -1])) for _ in range(target_rank)] for _ in range(n*m)]
            V_frac = [[Fraction(random.choice([0, 1, -1])) for _ in range(target_rank)] for _ in range(m*p)]
            
            # Solve for W
            W_frac = [[Fraction(0) for _ in range(target_rank)] for _ in range(p*n)]
            
            valid_solution = True
            
            for k in range(p):
                for l in range(n):
                    # System: sum_c W[k*n+l, c] * U[i*m+j, c] * V[j*p+k, c] = delta_{i,l}
                    # Unknowns: W[k*n+l, 0..R-1]
                    # Equations: for each i in [0, n), j in [0, m)
                    
                    A = []
                    b = []
                    for i in range(n):
                        for j in range(m):
                            row = []
                            for c in range(target_rank):
                                row.append(U_frac[i*m+j][c] * V_frac[j*p+k][c])
                            A.append(row)
                            b.append(Fraction(1) if i == l else Fraction(0))
                    
                    # Solve A * w = b for w
                    # A is n*m x target_rank (6 x R)
                    # We need a solution.
                    # Use Gaussian elimination with Fractions
                    try:
                        w = _solve_linear_system_exact(A, b)
                        if w is None:
                            valid_solution = False
                            break
                        W_frac[k*n+l] = w
                    except Exception:
                        valid_solution = False
                        break
                
                if not valid_solution:
                    break
            
            if valid_solution:
                # Verify the solution exactly
                U_list = [[float(x) for x in row] for row in U_frac]
                V_list = [[float(x) for x in row] for row in V_frac]
                W_list = [[float(x) for x in row] for row in W_frac]
                
                if verify_decomposition(U_list, V_list, W_list):
                    # Check if entries are "nice" (integers or half-integers)
                    # The problem allows integer/half-integer entries.
                    # Our solution might have larger denominators.
                    # We can try to scale or just return it if it passes verification.
                    # The prompt says "integer/half-integer entries".
                    # If the solution has other fractions, it might be invalid for the strict checker?
                    # "exactly decompose... with integer/half-integer entries"
                    # So we must ensure entries are in {k/2 | k in Z}.
                    
                    is_nice = True
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for val in row:
                                # Check if val * 2 is close to an integer
                                if abs(val * 2 - round(val * 2)) > 1e-9:
                                    is_nice = False
                                    break
                            if not is_nice:
                                break
                        if not is_nice:
                            break
                    
                    if is_nice:
                        best_rank = target_rank
                        best_U = np.array(U_list)
                        best_V = np.array(V_list)
                        best_W = np.array(W_list)
                        found = True
                        break # Found minimal rank for this attempt, move to next rank? No, break inner loop
                    else:
                        # Solution exists but not with nice numbers.
                        # We can try to find a nice solution by adding null space vectors?
                        # Or just reject and try another random init.
                        pass
            
        if found:
            break # Found a solution for this rank, stop searching higher ranks
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _solve_linear_system_exact(A, b):
    """
    Solve A * x = b where A is a list of lists of Fractions, b is a list of Fractions.
    Returns a list of Fractions x, or None if no solution.
    A is m x n. We want any solution.
    """
    m = len(A)
    n = len(A[0])
    
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian Elimination
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if M[row][col] != 0:
                pivot = row
                break
        
        if pivot == -1:
            continue
            
        # Swap rows
        M[rank], M[pivot] = M[pivot], M[rank]
        
        # Normalize pivot row
        pivot_val = M[rank][col]
        for c in range(col, n + 1):
            M[rank][c] /= pivot_val
            
        # Eliminate other rows
        for row in range(m):
            if row == rank:
                continue
            factor = M[row][col]
            if factor != 0:
                for c in range(col, n + 1):
                    M[row][c] -= factor * M[rank][c]
        
        rank += 1
        
    # Check for consistency
    for row in range(m):
        # If all coefficients are 0 but RHS is not 0, no solution
        all_zero = all(M[row][c] == 0 for c in range(n))
        if all_zero and M[row][n] != 0:
            return None
            
    # Find a solution
    # Set free variables to 0
    x = [Fraction(0) for _ in range(n)]
    
    # Identify pivot columns
    pivot_cols = []
    for col in range(n):
        is_pivot = False
        for row in range(rank):
            if M[row][col] == 1 and all(M[row][c] == 0 for c in range(n) if c != col):
                # This is a simplified check, better to track during elimination
                pass
        
    # Simpler: Back substitution
    # The matrix is in RREF.
    for i in range(rank):
        # Find the first non-zero column in this row
        pivot_col = -1
        for j in range(n):
            if M[i][j] != 0:
                pivot_col = j
                break
        if pivot_col == -1:
            continue
        # x[pivot_col] = M[i][n] - sum(M[i][j] * x[j] for j != pivot_col)
        # Since free vars are 0, x[pivot_col] = M[i][n]
        x[pivot_col] = M[i][n]
        
    return x

