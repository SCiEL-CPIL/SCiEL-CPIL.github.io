import time
import random
import numpy as np
from fractions import Fraction
import scipy.linalg

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    start_time = time.time()
    deadline = 110.0
    
    def is_valid_solution(U, V, W, r):
        # Check if U, V, W produce the target tensor
        # Target is all 1s for i in n, j in m, k in p
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def solve_for_W(U, V, r):
        # Build linear system for W
        # Variables x_{k,i,c}
        # Equation for (i,j,k): sum_c U[i*m+j][c] * V[j*p+k][c] * x_{k,i,c} = 1
        
        # Num vars: p * n * r
        # Num equations: n * m * p
        num_vars = p * n * r
        num_eqs = n * m * p
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        val = U[i*m+j][c] * V[j*p+k][c]
                        if val != 0:
                            var_idx = (k * n + i) * r + c
                            A[eq_idx, var_idx] = val
                    eq_idx += 1
        
        # Solve A x = b
        # Since it's underdetermined (usually), we use lstsq
        # We want a sparse solution ideally, but lstsq gives min-norm
        try:
            x, residuals, rank, sv = scipy.linalg.lstsq(A, b, cond=None)
            if residuals is None or residuals[0] > 1e-10:
                return None
            # Check if solution is "nice" (small rationals)
            # For now, just return the float solution and let the caller check
            return x
        except Exception:
            return None

    # Search for lower rank
    for r in range(rank_school - 1, 20, -1): # Try 23, 22, 21, 20
        if time.time() > deadline:
            break
            
        max_trials = 1000
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            # Bias towards sparse?
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
            
            # Solve for W
            x = solve_for_W(U, V, r)
            if x is None:
                continue
                
            # Reshape x to W
            W = [[0.0] * r for _ in range(p * n)]
            for ki in range(p * n):
                for c in range(r):
                    W[ki][c] = x[ki * r + c]
                    
            # Check if W has small rational values
            # We can try to snap to nearest integer or half-integer
            W_snapped = [[round(val * 2) / 2 for val in row] for row in W]
            
            # Verify with Fractions
            W_frac = [[Fraction(val).limit_denominator(100) for val in row] for row in W_snapped]
            U_frac = [[Fraction(val) for val in row] for row in U]
            V_frac = [[Fraction(val) for val in row] for row in V]
            
            if is_valid_solution(U_frac, V_frac, W_frac, r):
                # Found a valid solution
                best_U = U_frac
                best_V = V_frac
                best_W = W_frac
                best_rank = r
                # We can break early if we find a very low rank, 
                # but let's try to go lower if time permits
                # For now, break inner loop
                break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
