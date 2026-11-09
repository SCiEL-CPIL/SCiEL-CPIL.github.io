"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    The output factor follows W[k*n+i, r] convention.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies: for all i,j,k:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We use a random search with rational reconstruction to find low-rank decompositions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # The schoolbook rank is n*m*p = 60
    # Lower bounds: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # We'll try to find decompositions with rank < 60
    
    # First, let's try to find a good decomposition using random initialization
    # and rational approximation
    
    best_rank = n * m * p
    best_result = None
    
    # Start with schoolbook as baseline
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try random searches for lower rank decompositions
    # Use rational numbers with small denominators
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, rank):
        """Fast verification using numpy for floating point check first."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Try to find decompositions with rank r for r from lower_bound to schoolbook
    # Use random rational entries
    
    def try_rank(r, max_trials=2000):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Initialize with random small rationals
            # Use a mix of 0, +/-1, +/-1/2, etc.
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # Randomly assign values
            for idx in range(n * m):
                for ridx in range(r):
                    if random.random() < 0.3:  # 30% non-zero
                        val = random.choice([1, -1, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3)])
                        U[idx][ridx] = val
            
            for idx in range(m * p):
                for ridx in range(r):
                    if random.random() < 0.3:
                        val = random.choice([1, -1, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3)])
                        V[idx][ridx] = val
            
            for idx in range(p * n):
                for ridx in range(r):
                    if random.random() < 0.3:
                        val = random.choice([1, -1, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3)])
                        W[idx][ridx] = val
            
            # Verify
            if verify_decomposition(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 30 (lower bound) up to 59
    for r in range(30, n * m * p):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_trials=500)
        if result is not None:
            best_rank = r
            best_result = result
            # Found a valid decomposition, try to go lower
            continue
        else:
            # If we can't find rank r, it's unlikely we can find rank r+1 either
            # (this is a heuristic; in general it's not monotonic in this way for random search)
            # But for now, let's continue trying
            pass
    
    # If no improvement found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
