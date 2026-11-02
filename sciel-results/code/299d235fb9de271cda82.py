import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic (entries are half-integers)."""
    r = len(U[0])
    # Multiply by 2 to work with integers
    U2 = [[2 * U[i][j] for j in range(r)] for i in range(n*m)]
    V2 = [[2 * V[i][j] for j in range(r)] for i in range(m*p)]
    W2 = [[2 * W[i][j] for j in range(r)] for i in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i*m+j][c] * V2[j*p+k][c] * W2[k*n+i][c]
                if s != 8:  # 2*2*2=8
                    return False
    return True

def _solve_linear_system(A, b):
    """
    Solve Ax = b over rationals using Gaussian elimination.
    A is a list of lists (rows), b is a list.
    Returns solution as list of Fractions, or None if no solution.
    """
    n = len(A)
    m = len(A[0])
    # Augmented matrix
    M = [[Fraction(A[i][j]) for j in range(m)] + [Fraction(b[i])] for i in range(n)]
    
    # Gaussian elimination with partial pivoting
    for col in range(min(n, m)):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, m + 1):
                M[row][j] -= factor * M[col][j]
    
    # Back substitution
    x = [Fraction(0)] * m
    for i in range(n - 1, -1, -1):
        if M[i][i] == 0:
            continue
        s = M[i][m]
        for j in range(i + 1, m):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    
    # Verify solution
    for i in range(n):
        s = sum(M[i][j] * x[j] for j in range(m))
        if s != M[i][m]:
            return None
    
    return x

def _try_rank_alternating(n, m, p, r, max_iter=50, timeout=10):
    """
    Try to find a rank r decomposition using alternating linear solves with rational arithmetic.
    Works with integer-scaled variables (multiply by 2 for half-integers).
    """
    start_time = time.time()
    
    # Initialize with schoolbook-like pattern
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Initialize: use first r columns of schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < r:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                else:
                    break
            if col >= r:
                break
        if col >= r:
            break
    
    # If r < n*m*p, we need to be smarter
    # Let's initialize with a pattern that might work
    if r < n * m * p:
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Simple initialization: spread ones
        for c in range(r):
            # Assign to some indices
            u_idx = c % (n * m)
            v_idx = c % (m * p)
            w_idx = c % (p * n)
            U[u_idx][c] = 1
            V[v_idx][c] = 1
            W[w_idx][c] = 1
    
    for iteration in range(max_iter):
        if time.time() - start_time > timeout:
            return None
        
        # Step 1: Fix V and W, solve for U
        # For each (i,j): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This gives p equations for each (i,j), but they're coupled through c
        # Actually, for fixed (i,j), the equation is:
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for each k
        # This is a linear system in U[i*m+j, :] with p equations and r unknowns
        
        U_new = [[0] * r for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                # Build linear system: A * U_row = b
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # b[k] = 1
                A = [[V[j*p+k][c] * W[k*n+i][c] for c in range(r)] for k in range(p)]
                b = [1] * p
                sol = _solve_linear_system(A, b)
                if sol is None:
                    return None
                U_new[i*m+j] = [int(sol[c]) for c in range(r)]
        
        # Step 2: Fix U and W, solve for V
        V_new = [[0] * r for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # b[i] = 1
                A = [[U[i*m+j][c] * W[k*n+i][c] for c in range(r)] for i in range(n)]
                b = [1] * n
                sol = _solve_linear_system(A, b)
                if sol is None:
                    return None
                V_new[j*p+k] = [int(sol[c]) for c in range(r)]
        
        # Step 3: Fix U and V, solve for W
        W_new = [[0] * r for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # b[j] = 1
                A = [[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)]
                b = [1] * m
                sol = _solve_linear_system(A, b)
                if sol is None:
                    return None
                W_new[k*n+i] = [int(sol[c]) for c in range(r)]
        
        # Check convergence
        U, V, W = U_new, V_new, W_new
        
        # Verify
        if _verify_exact(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_rank_search(n, m, p, r, timeout=30):
    """
    Search for rank r decomposition by trying different initializations.
    """
    start_time = time.time()
    
    # Try multiple random integer initializations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small integers
        U = [[np.random.randint(-2, 3) for _ in range(r)] for _ in range(n * m)]
        V = [[np.random.randint(-2, 3) for _ in range(r)] for _ in range(m * p)]
        W = [[np.random.randint(-2, 3) for _ in range(r)] for _ in range(p * n)]
        
        # Run alternating method
        result = _try_rank_alternating(n, m, p, r, max_iter=30, timeout=timeout/20)
        if result:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But tensor rank can be lower than this in some cases
    # For matrix multiplication, known bounds suggest rank around n*m*p/2 to n*m*p
    
    best_result = None
    
    # First, get schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank
    for r in range(10, rank_school):
        if time.time() > deadline:
            break
        
        result = _try_rank_search(n, m, p, r, timeout=min(30, deadline - time.time()))
        if result:
            best_result = result
            break  # Found a valid decomposition, return it
    
    return best_result
