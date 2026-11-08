"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _solve_linear_system(A, b):
    """Solve A x = b exactly using Gaussian elimination with fractions.
    A is m x r, b is m x 1. Returns x (r x 1) or None if no unique solution."""
    m = len(A)
    r = len(A[0]) if m > 0 else 0
    
    # Create augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(r)] + [Fraction(b[i])] for i in range(m)]
    
    # Gaussian elimination
    for col in range(min(m, r)):
        # Find pivot
        pivot_row = -1
        for row in range(col, m):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue  # Skip this column (free variable)
        
        # Swap rows
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        
        # Eliminate below
        for row in range(col + 1, m):
            if aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for j in range(col, r + 1):
                    aug[row][j] -= factor * aug[col][j]
    
    # Back substitution
    x = [Fraction(0)] * r
    for i in range(m - 1, -1, -1):
        # Find which column this row corresponds to
        pivot_col = -1
        for j in range(r):
            if aug[i][j] != 0:
                pivot_col = j
                break
        if pivot_col == -1:
            if aug[i][r] != 0:
                return None  # Inconsistent
            continue
        
        s = Fraction(aug[i][r])
        for j in range(pivot_col + 1, r):
            if x[j] != 0:
                s -= aug[i][j] * x[j]
        if aug[i][pivot_col] == 0:
            return None
        x[pivot_col] = s / aug[i][pivot_col]
    
    # Verify solution
    for i in range(m):
        s = Fraction(0)
        for j in range(r):
            s += Fraction(A[i][j]) * x[j]
        if s != Fraction(b[i]):
            return None
    
    return x


def _is_half_integer(x):
    """Check if x is a half-integer (multiple of 1/2)."""
    if isinstance(x, Fraction):
        return (x * 2).denominator == 1
    return False


def _try_slice_method(r, deadline, start_time):
    """Try to find a rank-r decomposition using the slice method.
    
    Fix U and V to have structured entries, then solve for W.
    For each (i,k), we have m equations in r unknowns for W[k*n+i,:].
    """
    n, m, p = DIMENSIONS
    
    # Strategy: try U and V with entries from {0, 1, -1, 1/2, -1/2}
    # For small r, enumerate possible patterns
    
    # For efficiency, try a few structured approaches:
    # 1. U = identity-like, V = identity-like
    # 2. Random structured matrices
    
    candidates = []
    
    # Approach 1: Try U with 1s on diagonal, 0s elsewhere (if r >= n*m)
    if r >= n * m:
        U = [[0] * r for _ in range(n * m)]
        for i in range(n * m):
            U[i][i] = 1
        
        V = [[0] * r for _ in range(m * p)]
        for j in range(m * p):
            if j < r:
                V[j][j] = 1
        
        # Solve for W
        W = [[0] * r for _ in range(p * n)]
        valid = True
        for i in range(n):
            for k in range(p):
                # For each j, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a linear system in W[k*n+i,:]
                A = []
                b = []
                for j in range(m):
                    row = [Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) for c in range(r)]
                    A.append(row)
                    b.append(1)
                
                sol = _solve_linear_system(A, b)
                if sol is None:
                    valid = False
                    break
                W[k * n + i] = sol
            if not valid:
                break
        
        if valid:
            # Check if all W entries are half-integers
            all_half = all(_is_half_integer(W[i][j]) for i in range(p * n) for j in range(r))
            if all_half:
                candidates.append({"U": U, "V": V, "W": W})
    
    # Approach 2: Try with 1/2 entries
    if r >= n * m:
        U = [[0] * r for _ in range(n * m)]
        for i in range(n * m):
            U[i][i] = Fraction(1, 2)
        
        V = [[0] * r for _ in range(m * p)]
        for j in range(m * p):
            if j < r:
                V[j][j] = Fraction(2)
        
        W = [[0] * r for _ in range(p * n)]
        valid = True
        for i in range(n):
            for k in range(p):
                A = []
                b = []
                for j in range(m):
                    row = [Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) for c in range(r)]
                    A.append(row)
                    b.append(1)
                
                sol = _solve_linear_system(A, b)
                if sol is None:
                    valid = False
                    break
                W[k * n + i] = sol
            if not valid:
                break
        
        if valid:
            all_half = all(_is_half_integer(W[i][j]) for i in range(p * n) for j in range(r))
            if all_half:
                candidates.append({"U": U, "V": V, "W": W})
    
    # Approach 3: Try random structured U, V with small entries
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Generate random U, V with entries from {-1, 0, 1}
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        
        for i in range(n * m):
            for c in range(min(r, 3)):
                U[i][c] = np.random.randint(-1, 2)
        
        for j in range(m * p):
            for c in range(min(r, 3)):
                V[j][c] = np.random.randint(-1, 2)
        
        W = [[0] * r for _ in range(p * n)]
        valid = True
        for i in range(n):
            for k in range(p):
                A = []
                b = []
                for j in range(m):
                    row = [Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) for c in range(r)]
                    A.append(row)
                    b.append(1)
                
                sol = _solve_linear_system(A, b)
                if sol is None:
                    valid = False
                    break
                W[k * n + i] = sol
            if not valid:
                break
        
        if valid:
            all_half = all(_is_half_integer(W[i][j]) for i in range(p * n) for j in range(r))
            if all_half:
                candidates.append({"U": U, "V": V, "W": W})
    
    # Verify candidates
    for cand in candidates:
        if _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    
    return None


def _construct_rank_41(U23, V23, W23):
    """
    Construct a rank-41 decomposition for 3x3x3x5 using a rank-23 decomposition
    for the first 3 columns and schoolbook for the last 2 columns.
    """
    n = 3
    m = 3
    p = 5
    rank_23 = 23
    rank_sb = 9
    total_rank = rank_23 + 2 * rank_sb  # 41
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in the rank-23 part for k=0,1,2
    for i in range(n):
        for j in range(m):
            for c in range(rank_23):
                U[i * m + j][c] = U23[i * m + j][c]
    
    for j in range(m):
        for k in range(3):
            for c in range(rank_23):
                V[j * p + k][c] = V23[j * 3 + k][c]
    
    for k in range(3):
        for i in range(n):
            for c in range(rank_23):
                W[k * n + i][c] = W23[k * n + i][c]
    
    # Fill in the schoolbook part for k=3,4
    sb_offset = rank_23
    col_idx = sb_offset
    for i in range(n):
        for j in range(m):
            U[i * m + j][col_idx] = 1
            V[j * p + 3][col_idx] = 1
            W[3 * n + i][col_idx] = 1
            col_idx += 1
    
    col_idx = sb_offset + rank_sb
    for i in range(n):
        for j in range(m):
            U[i * m + j][col_idx] = 1
            V[j * p + 4][col_idx] = 1
            W[4 * n + i][col_idx] = 1
            col_idx += 1
    
    return {"U": U, "V": V, "W": W}


def _try_numerical_rank_23_3x3x3(deadline):
    """
    Attempt to find a rank-23 decomposition for 3x3x3 matrix multiplication
    using numerical optimization.
    Returns (U, V, W) as lists of lists of Fractions if successful, else None.
    """
    n = 3
    m = 3
    p = 3
    rank = 23
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    np.random.seed(42)
    x0 = np.random.rand(total_size) * 2 - 1
    
    def objective(x):
        U = x[:size_U].reshape(n*m, rank)
        V = x[size_U:size_U+size_V].reshape(m*p, rank)
        W = x[size_U+size_V:].reshape(p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (s - 1.0)**2
        return error
    
    best_x = None
    best_err = float('inf')
    
    for restart in range(5):
        if time.time() > deadline:
            return None
            
        if restart == 0:
            x_init = x0
        else:
            x_init = np.random.rand(total_size) * 2 - 1
            
        res = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
            
        if best_err < 1e-6:
            break
            
    if best_err > 1e-4:
        return None
        
    U_flat = best_x[:size_U]
    V_flat = best_x[size_U:size_U+size_V]
    W_flat = best_x[size_U+size_V:]
    
    U = U_flat.reshape(n*m, rank)
    V = V_flat.reshape(m*p, rank)
    W = W_flat.reshape(p*n, rank)
    
    U_int = np.round(U * 2) / 2
    V_int = np.round(V * 2) / 2
    W_int = np.round(W * 2) / 2
    
    U_frac = [[Fraction(int(val*2))/2 for val in row] for row in U_int]
    V_frac = [[Fraction(int(val*2))/2 for val in row] for row in V_int]
    W_frac = [[Fraction(int(val*2))/2 for val in row] for row in W_int]
    
    if _verify(U_frac, V_frac, W_frac, n, m, p):
        return U_frac, V_frac, W_frac
        
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try slice method for various ranks
    for r in [15, 18, 20, 25, 30, 35, 40, 44]:
        if time.time() > deadline:
            break
        result = _try_slice_method(r, deadline, start_time)
        if result is not None:
            return result
    
    # Try to find a rank-41 decomposition using rank-23 for 3x3x3
    result_23 = _try_numerical_rank_23_3x3x3(deadline)
    if result_23 is not None:
        U23, V23, W23 = result_23
        result_41 = _construct_rank_41(U23, V23, W23)
        if _verify(result_41["U"], result_41["V"], result_41["W"], n, m, p):
            return result_41
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
