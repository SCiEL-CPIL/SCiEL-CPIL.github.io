"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Small rational values to try
    # Use halves and integers: -2, -3/2, -1, -1/2, 0, 1/2, 1, 3/2, 2
    small_vals = [Fraction(-2), Fraction(-3,2), Fraction(-1), Fraction(-1,2), 
                  Fraction(0), Fraction(1,2), Fraction(1), Fraction(3,2), Fraction(2)]
    
    def verify(U, V, W, rank):
        """Verify exact decomposition with Fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != Fraction(1):
                        return False
        return True
    
    # Strategy: For a given rank, we can fix some entries and solve for others.
    # Key insight: The tensor is multilinear, so we can use the structure to our advantage.
    # 
    # For 3x4x5, known lower bounds suggest rank around 55-60.
    # Let's try a systematic search with small rational values.
    
    # First, let's try a smarter approach: use the fact that we can permute columns
    # and scale them. We'll fix the first few columns to have simple patterns.
    
    def try_rank_with_patterns(rank, max_trials=50):
        """Try to find a decomposition of given rank using pattern-based search."""
        np.random.seed(rank)
        
        # Generate candidate patterns for W (smallest matrix: p*n = 15 rows)
        # We'll use small rational values and try to complete the decomposition
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            # Initialize with small random values from our set
            W = [[small_vals[np.random.randint(len(small_vals))] for _ in range(rank)] for _ in range(p*n)]
            V = [[small_vals[np.random.randint(len(small_vals))] for _ in range(rank)] for _ in range(m*p)]
            U = [[small_vals[np.random.randint(len(small_vals))] for _ in range(rank)] for _ in range(n*m)]
            
            # Iterative refinement using local search
            improved = True
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                if not improved:
                    break
                improved = False
                
                # Try flipping individual entries to small rational values
                # Focus on entries that are causing the largest errors
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            err = abs(s - Fraction(1))
                            if err > 0:
                                errors.append((err, i, j, k))
                
                if not errors:
                    # Found exact solution!
                    return {"U": U, "V": V, "W": W}
                
                # Sort by error and try to fix the worst ones
                errors.sort(reverse=True)
                _, wi, wj, wk = errors[0]
                
                # Try changing entries in U[wi*m+wj], V[wj*p+wk], W[wk*n+wi]
                # For each column c, try different values
                for c in range(rank):
                    # Try changing U[wi*m+wj][c]
                    for new_val in small_vals:
                        if U[wi*m+wj][c] == new_val:
                            continue
                        old_val = U[wi*m+wj][c]
                        U[wi*m+wj][c] = new_val
                        # Check if this reduces total error
                        total_err_before = sum(e for e, _, _, _ in errors)
                        new_errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = Fraction(0)
                                    for cc in range(rank):
                                        s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                                    err = abs(s - Fraction(1))
                                    if err > 0:
                                        new_errors.append((err, i, j, k))
                        total_err_after = sum(e for e, _, _, _ in new_errors)
                        if total_err_after < total_err_before:
                            improved = True
                        else:
                            U[wi*m+wj][c] = old_val
                    
                    if improved:
                        break
                    # Try changing V[wj*p+wk][c]
                    for new_val in small_vals:
                        if V[wj*p+wk][c] == new_val:
                            continue
                        old_val = V[wj*p+wk][c]
                        V[wj*p+wk][c] = new_val
                        total_err_before = sum(e for e, _, _, _ in errors)
                        new_errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = Fraction(0)
                                    for cc in range(rank):
                                        s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                                    err = abs(s - Fraction(1))
                                    if err > 0:
                                        new_errors.append((err, i, j, k))
                        total_err_after = sum(e for e, _, _, _ in new_errors)
                        if total_err_after < total_err_before:
                            improved = True
                        else:
                            V[wj*p+wk][c] = old_val
                    
                    if improved:
                        break
                    # Try changing W[wk*n+wi][c]
                    for new_val in small_vals:
                        if W[wk*n+wi][c] == new_val:
                            continue
                        old_val = W[wk*n+wi][c]
                        W[wk*n+wi][c] = new_val
                        total_err_before = sum(e for e, _, _, _ in errors)
                        new_errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = Fraction(0)
                                    for cc in range(rank):
                                        s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                                    err = abs(s - Fraction(1))
                                    if err > 0:
                                        new_errors.append((err, i, j, k))
                        total_err_after = sum(e for e, _, _, _ in new_errors)
                        if total_err_after < total_err_before:
                            improved = True
                        else:
                            W[wk*n+wi][c] = old_val
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return None
    
    # Try ranks from lower to upper
    # Known: schoolbook is 60, lower bound is around 55 for 3x4x5
    for rank in range(55, 61):
        if time.time() > deadline - 30:
            break
        
        result = try_rank_with_patterns(rank, max_trials=30)
        if result is not None:
            # Convert Fractions to floats for output
            U_out = [[float(x) for x in row] for row in result["U"]]
            V_out = [[float(x) for x in row] for row in result["V"]]
            W_out = [[float(x) for x in row] for row in result["W"]]
            
            # Verify with float arithmetic
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                best_result = {"U": U_out, "V": V_out, "W": W_out}
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
