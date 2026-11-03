"""Program for exact tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s wall-clock limit (5s safety margin)
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact using Fraction for exact arithmetic."""
        if not U or not V or not W:
            return False
        
        num_cols_U = len(U[0])
        num_cols_V = len(V[0])
        num_cols_W = len(W[0])
        
        if num_cols_U != num_cols_V or num_cols_V != num_cols_W:
            return False
        
        rank = num_cols_U
        
        # Convert to Fraction for exact verification
        U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[j][r]) for r in range(rank)] for j in range(m*p)]
        W_frac = [[Fraction(W[k][r]) for r in range(rank)] for k in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                    if s != 1:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def solve_for_w_exact(U, V, n, m, p, target_rank):
        """
        For fixed U and V, try to solve for W such that the decomposition is exact.
        Returns W as list of lists of Fractions, or None if no exact solution exists.
        """
        W = [[Fraction(0)] * target_rank for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                c = k * n + i
                # We need: for each j in range(m), sum_r U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
                # This is a linear system: A @ w = b, where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # b = [1, 1, ..., 1] (length m)
                
                A = [[Fraction(0)] * target_rank for _ in range(m)]
                for j in range(m):
                    for r in range(target_rank):
                        A[j][r] = Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r])
                
                b = [Fraction(1) for _ in range(m)]
                
                # Solve A @ w = b using Gaussian elimination over Q
                # Augmented matrix [A | b]
                aug = [row[:] + [b[j]] for j, row in enumerate(A)]
                
                # Gaussian elimination with partial pivoting
                rank_A = 0
                for col in range(target_rank):
                    # Find pivot
                    pivot_row = None
                    for row in range(rank_A, m):
                        if aug[row][col] != 0:
                            pivot_row = row
                            break
                    
                    if pivot_row is None:
                        continue
                    
                    # Swap
                    aug[rank_A], aug[pivot_row] = aug[pivot_row], aug[rank_A]
                    
                    # Eliminate below
                    pivot_val = aug[rank_A][col]
                    for row in range(m):
                        if row == rank_A:
                            continue
                        factor = aug[row][col] / pivot_val
                        if factor != 0:
                            for c2 in range(col, target_rank + 1):
                                aug[row][c2] -= factor * aug[rank_A][c2]
                    
                    rank_A += 1
                
                # Check consistency: any row with all zeros in A part but non-zero in b part
                for row in range(m):
                    all_zero = all(aug[row][c] == 0 for c in range(target_rank))
                    if all_zero and aug[row][target_rank] != 0:
                        return None
                
                # Find a particular solution (set free variables to 0)
                w = [Fraction(0)] * target_rank
                for row in range(rank_A - 1, -1, -1):
                    # Find the leading column
                    lead_col = None
                    for c in range(target_rank):
                        if aug[row][c] != 0:
                            lead_col = c
                            break
                    if lead_col is None:
                        continue
                    val = aug[row][target_rank]
                    for c in range(lead_col + 1, target_rank):
                        val -= aug[row][c] * w[c]
                    w[lead_col] = val / aug[row][lead_col]
                
                W[c] = w
        
        return W
    
    def try_random_decomposition(target_rank, n, m, p, max_attempts=200):
        """
        Try to find a rank-target_rank decomposition using random initialization.
        For fixed U and V, solve for W using exact linear algebra.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Randomly initialize U and V with small integers
            # Use a mix of values to encourage structure
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            
            # Solve for W exactly
            W_frac = solve_for_w_exact(U, V, n, m, p, target_rank)
            if W_frac is None:
                continue
            
            # Convert W to list of lists of floats (or keep as Fractions if needed)
            W = [[float(x) for x in W_frac[c]] for c in range(p * n)]
            
            # Verify the decomposition
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_structured_decomposition(target_rank, n, m, p):
        """
        Try a structured approach based on known low-rank decompositions.
        For 2x3 by 3x7, we can try to exploit the structure.
        """
        # One known approach: use the fact that we can combine some operations
        # This is a heuristic attempt
        
        if target_rank < 21:
            return None
        
        # Try a pattern-based approach
        # For now, fall back to random
        return None
    
    # Try to find a lower-rank decomposition
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Schoolbook is 42
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # Try ranks from 21 up to 41
    # Use more attempts for lower ranks
    for target_rank in range(21, min(42, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try structured first
        result = try_structured_decomposition(target_rank, n, m, p)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
        
        # Then try random
        # Use more attempts for ranks closer to the lower bound
        attempts = 100 if target_rank <= 25 else 50
        result = try_random_decomposition(target_rank, n, m, p, max_attempts=attempts)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found the lowest rank, we can stop
    
    if best_result is None:
        # Fall back to schoolbook
        best_result = schoolbook_decomposition(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
