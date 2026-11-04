"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from fractions import Fraction
from itertools import product

def _schoolbook_solution():
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

def _verify_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _check_half_integer_solution(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _solve_linear_system_exact(A, b):
    """Solve A*x = b over rationals using Gaussian elimination with Fractions.
    Returns solution vector as list of Fractions, or None if no solution."""
    n = len(A)
    m = len(A[0]) if A else 0
    if n == 0:
        return [Fraction(0)] * m
    # Augmented matrix
    aug = [[Fraction(A[i][j]) for j in range(m)] + [Fraction(b[i])] for i in range(n)]
    
    # Gaussian elimination
    row = 0
    for col in range(m):
        # Find pivot
        pivot = None
        for r in range(row, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        aug[row], aug[pivot] = aug[pivot], aug[row]
        # Normalize
        pivot_val = aug[row][col]
        for j in range(col, m + 1):
            aug[row][j] /= pivot_val
        # Eliminate
        for r in range(n):
            if r != row and aug[r][col] != 0:
                factor = aug[r][col]
                for j in range(col, m + 1):
                    aug[r][j] -= factor * aug[row][j]
        row += 1
        if row == n:
            break
    
    # Check for inconsistency
    for r in range(row, n):
        if all(aug[r][j] == 0 for j in range(m)) and aug[r][m] != 0:
            return None
    
    # Extract solution (for underdetermined, set free variables to 0)
    x = [Fraction(0)] * m
    for i in range(row):
        for j in range(m):
            if aug[i][j] == 1:
                x[j] = aug[i][m]
                break
    return x

def _try_rank_r_construction(r, n, m, p, seed=None, time_limit=None):
    """Try to find rank-r decomposition using structured support patterns."""
    if seed is not None:
        random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    num_eq = n * m * p
    
    # Strategy: For each rank column c, choose a support pattern
    # U_support_c: subset of (i,j) pairs
    # V_support_c: subset of (j,k) pairs  
    # W_support_c: subset of (k,i) pairs
    # The product U[i,j,c]*V[j,k,c]*W[k,i,c] contributes to equation (i,j,k)
    # only if (i,j) in U_support, (j,k) in V_support, (k,i) in W_support
    
    # For the system to be solvable, each equation (i,j,k) must be covered by
    # at least one column c where all three supports include the relevant indices.
    
    # Approach: Try small support sizes and random patterns
    max_attempts = 200
    for attempt in range(max_attempts):
        if time_limit and time.time() > time_limit:
            break
        
        # Generate random support patterns
        U_sup = []
        V_sup = []
        W_sup = []
        
        for c in range(r):
            # Random subsets
            u_nz = random.randint(1, min(6, nm))
            v_nz = random.randint(1, min(6, mp))
            w_nz = random.randint(1, min(6, pn))
            
            u_set = set(random.sample(range(nm), u_nz))
            v_set = set(random.sample(range(mp), v_nz))
            w_set = set(random.sample(range(pn), w_nz))
            
            U_sup.append(u_set)
            V_sup.append(v_set)
            W_sup.append(w_set)
        
        # Build the linear system for coefficients
        # Variables: for each c, for each (i,j) in U_sup[c]: u_c[i,j]
        #            for each (j,k) in V_sup[c]: v_c[j,k]
        #            for each (k,i) in W_sup[c]: w_c[k,i]
        # Equation (i,j,k): sum over c where (i*m+j in U_sup[c], j*p+k in V_sup[c], k*n+i in W_sup[c])
        #                    of u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
        
        # This is still a 3-linear system. Let's fix U and V, solve for W.
        
        # Assign random values to U and V (small integers)
        U_vals = {}
        V_vals = {}
        for c in range(r):
            for idx in U_sup[c]:
                U_vals[(c, idx)] = random.choice([-1, 1])
            for idx in V_sup[c]:
                V_vals[(c, idx)] = random.choice([-1, 1])
        
        # For each (k,i), build linear system for W
        # Equation: for each j, sum_c U_vals[(c, i*m+j)] * V_vals[(c, j*p+k)] * w_c[k*n+i] = 1
        # where w_c[k*n+i] is non-zero only if k*n+i in W_sup[c]
        
        W_vals = {}
        success = True
        
        for k in range(p):
            for i in range(n):
                # Build m x (number of free W variables) system
                # Free variables: w_c[k*n+i] for c where k*n+i in W_sup[c]
                free_vars = []
                for c in range(r):
                    if k * n + i in W_sup[c]:
                        free_vars.append(c)
                
                if not free_vars:
                    # No variable can satisfy this equation - fail
                    success = False
                    break
                
                # Build A matrix: m rows, len(free_vars) columns
                A = [[0] * len(free_vars) for _ in range(m)]
                b = [1] * m
                
                for j in range(m):
                    for vi, c in enumerate(free_vars):
                        u_val = U_vals.get((c, i * m + j), 0)
                        v_val = V_vals.get((c, j * p + k), 0)
                        A[j][vi] = u_val * v_val
                
                # Solve A * x = b
                if len(free_vars) < m:
                    # Underdetermined - use least squares
                    try:
                        A_np = np.array(A, dtype=float)
                        b_np = np.array(b, dtype=float)
                        x, residuals, rank_a, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        # Check if close to half-integers
                        x_rounded = np.round(x * 2) / 2
                        if np.allclose(x, x_rounded, atol=1e-6):
                            for vi, c in enumerate(free_vars):
                                W_vals[(c, k * n + i)] = float(x_rounded[vi])
                        else:
                            success = False
                            break
                    except:
                        success = False
                        break
                else:
                    # Square or overdetermined - solve exactly
                    x = _solve_linear_system_exact(A, b)
                    if x is None:
                        success = False
                        break
                    # Check if all are half-integers
                    valid = True
                    for vi, val in enumerate(x):
                        if abs(val * 2 - round(val * 2)) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        success = False
                        break
                    for vi, c in enumerate(free_vars):
                        W_vals[(c, k * n + i)] = float(x[vi])
                
                if not success:
                    break
            
            if not success:
                break
        
        if not success:
            continue
        
        # Build U, V, W matrices
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        
        for (c, idx), val in U_vals.items():
            U[idx][c] = val
        for (c, idx), val in V_vals.items():
            V[idx][c] = val
        for (c, idx), val in W_vals.items():
            W[idx][c] = val
        
        # Verify
        if _verify_decomposition(U, V, W) and _check_half_integer_solution(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook_solution()
    best_rank = len(best_result["U"][0])
    
    # Try ranks from lower bound up
    lower_bound = max(n, m, p)
    
    for r in range(lower_bound, min(best_rank, 40)):
        if time.time() > deadline:
            break
        
        result = _try_rank_r_construction(r, n, m, p, seed=r * 1000, time_limit=deadline)
        if result is not None:
            return result
    
    return best_result
# EVOLVE-BLOCK-END
