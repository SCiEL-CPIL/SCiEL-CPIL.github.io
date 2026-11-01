"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

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

def _try_exact_search(n, m, p, time_budget, max_rank=55):
    """
    Systematic exact search: try small integer/half-integer entries in U and V,
    solve for W using exact rational arithmetic, and verify.
    Exploits n=2 to reduce the search space.
    """
    deadline = time.time() + time_budget
    rng = random.Random(42)
    
    best = None
    best_rank = n * m * p + 1
    
    # Candidate values for entries (integers and half-integers)
    candidate_vals = [-2, -1, Fraction(0), 1, 2]
    
    # Try ranks from lower bound up to max_rank
    # Lower bound: for n*m*p tensor, rank >= max(n*m, m*p, p*n) but more carefully:
    # rank >= max(n*m, m*p, p*n) is not quite right; use rank >= n*m*p / something
    # Actually rank >= max(n*m, m*p, p*n) is a valid lower bound
    lb = max(n*m, m*p, p*n)
    start_rank = max(25, lb)
    
    for r in range(start_rank, min(max_rank + 1, n*m*p)):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        # For each rank r, try random assignments of small values to U and V
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Randomly assign small values to U (n*m rows) and V (m*p rows)
            U = [[Fraction(0)] * r for _ in range(n*m)]
            V = [[Fraction(0)] * r for _ in range(m*p)]
            
            # Fill U with sparse random small values
            for i in range(n*m):
                for c in range(r):
                    if rng.random() < 0.5:
                        U[i][c] = Fraction(rng.choice([-2, -1, 1, 2]))
            
            # Fill V with sparse random small values
            for j in range(m*p):
                for c in range(r):
                    if rng.random() < 0.5:
                        V[j][c] = Fraction(rng.choice([-2, -1, 1, 2]))
            
            # For each (i,k) pair, solve for W[k*n+i] using exact arithmetic
            # We need: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1 for all j
            # This is a linear system in W[k*n+i][c]
            W = [[Fraction(0)] * r for _ in range(p*n)]
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    # Build the linear system: for each j, sum_c (U[i*m+j][c] * V[j*p+k][c]) * W[k*n+i][c] = 1
                    # This is A @ w = b where A[j][c] = U[i*m+j][c] * V[j*p+k][c], b = ones
                    A = [[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)]
                    b = [Fraction(1)] * m
                    
                    # Check if the system is consistent using exact Gaussian elimination
                    # Augmented matrix [A | b]
                    aug = [row[:] + [b[j]] for j, row in enumerate(A)]
                    
                    # Gaussian elimination over Fractions
                    n_rows = m
                    n_cols = r
                    rank_aug = 0
                    for col in range(n_cols):
                        # Find pivot
                        pivot_row = -1
                        for row in range(rank_aug, n_rows):
                            if aug[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row == -1:
                            continue
                        
                        # Swap
                        aug[rank_aug], aug[pivot_row] = aug[pivot_row], aug[rank_aug]
                        
                        # Eliminate below
                        for row in range(rank_aug + 1, n_rows):
                            factor = aug[row][col] / aug[rank_aug][col]
                            for c in range(col, n_cols + 1):
                                aug[row][c] -= factor * aug[rank_aug][c]
                        
                        rank_aug += 1
                    
                    # Check consistency: no row with all zeros in A part but non-zero in b part
                    consistent = True
                    for row in range(rank_aug, n_rows):
                        if any(aug[row][c] != 0 for c in range(n_cols)):
                            consistent = False
                            break
                        if aug[row][n_cols] != 0:
                            consistent = False
                            break
                    if not consistent:
                        feasible = False
                        break
                    
                    # Find a solution (set free variables to 0, solve for pivot variables)
                    w = [Fraction(0)] * r
                    # Track which columns are pivot columns
                    pivot_cols = []
                    aug2 = [row[:] for row in aug]
                    rank2 = 0
                    for col in range(n_cols):
                        pivot_row = -1
                        for row in range(rank2, n_rows):
                            if aug2[row][col] != 0:
                                pivot_row = row
                                break
                        if pivot_row == -1:
                            continue
                        aug2[rank2], aug2[pivot_row] = aug2[pivot_row], aug2[rank2]
                        for row in range(rank2 + 1, n_rows):
                            factor = aug2[row][col] / aug2[rank2][col]
                            for c in range(col, n_cols + 1):
                                aug2[row][c] -= factor * aug2[rank2][c]
                        # Also eliminate above for RREF
                        for row in range(rank2):
                            factor = aug2[row][col] / aug2[rank2][col]
                            for c in range(col, n_cols + 1):
                                aug2[row][c] -= factor * aug2[rank2][c]
                        pivot_cols.append(col)
                        rank2 += 1
                    
                    # Back-substitute
                    for row in range(rank2):
                        col = pivot_cols[row]
                        w[col] = aug2[row][n_cols]
                    
                    W[k*n+i] = w
            
            if not feasible:
                continue
            
            # Verify the full decomposition
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if s != 1:
                            ok = False
                            break
            
            if ok:
                # Convert to list of floats/half-integers for output
                def frac_to_out(f):
                    return float(f)
                
                U_list = [[frac_to_out(v) for v in row] for row in U]
                V_list = [[frac_to_out(v) for v in row] for row in V]
                W_list = [[frac_to_out(v) for v in row] for row in W]
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    break
        
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try exact search
    time_left = deadline - time.time()
    if time_left > 20:
        result = _try_exact_search(n, m, p, time_left, max_rank=55)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
    
    return best
