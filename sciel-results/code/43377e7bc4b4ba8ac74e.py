"""Tensor decomposition search for 3x3 * 3x4 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
import itertools
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_school = n * m * p
    best_result = None
    best_rank = rank_school
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Target shapes
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    
    # Try ranks from low to high
    # Known lower bound: max(n*m, m*p, p*n) = max(9,12,12) = 12
    # But tensor rank is likely higher. Let's try from 13 upward.
    # For 3x3x3, best known is 23 (Strassen-like). For 3x3x4, it could be around 20-25.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U @ outer @ V @ outer @ W reconstructs T exactly."""
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check using the tensor product
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # Check exact match (with tolerance for floating point)
        if not np.allclose(T_rec, T, atol=1e-10, rtol=1e-10):
            return False
        return True
    
    def try_rank(rank, max_attempts=500):
        """Try to find a valid decomposition of given rank using random search."""
        nonlocal best_result, best_rank, start_time, deadline
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with rational-like values
            # Use small integers and half-integers for better chance of exactness
            # Strategy: random sparse matrices with entries from {-1, 0, 1, 0.5, -0.5}
            
            # Create random factor matrices
            def random_factor(dim, rank, sparsity=0.3):
                M = np.zeros((dim, rank))
                for i in range(dim):
                    for r in range(rank):
                        if random.random() < sparsity:
                            # Choose from a small set of values
                            val = random.choice([-1, 1, -0.5, 0.5, 2, -2])
                            M[i, r] = val
                return M
            
            U = random_factor(dim_U, rank, 0.4)
            V = random_factor(dim_V, rank, 0.4)
            W = random_factor(dim_W, rank, 0.4)
            
            if verify_decomposition(U, V, W, rank):
                # Convert to exact fractions
                U_frac = [[Fraction(U[i, r]).limit_denominator(1000) for r in range(rank)] for i in range(dim_U)]
                V_frac = [[Fraction(V[i, r]).limit_denominator(1000) for r in range(rank)] for i in range(dim_V)]
                W_frac = [[Fraction(W[i, r]).limit_denominator(1000) for r in range(rank)] for i in range(dim_W)]
                
                # Verify with exact arithmetic
                T_exact = np.zeros((dim_U, dim_V, dim_W), dtype=object)
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            s = Fraction(0)
                            for r in range(rank):
                                s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                            T_exact[a, b, c] = s
                
                # Check against target
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_a = i * m + j
                            idx_b = j * p + k
                            idx_c = k * n + i
                            if T_exact[idx_a, idx_b, idx_c] != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Also check all other entries are 0
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            found = False
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        if (i * m + j == a and j * p + k == b and k * n + i == c):
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if not found and T_exact[a, b, c] != 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U_frac, V_frac, W_frac, rank
        
        return None
    
    # Try structured approaches first
    # Approach 1: Try to find a decomposition using Strassen-like ideas
    # For 3x3x3, the best known rank is 23. For 3x3x4, we expect something similar or slightly higher.
    
    # Let's try ranks from 14 upward
    for rank in range(14, min(30, rank_school)):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, max_attempts=200)
        if result is not None:
            U_frac, V_frac, W_frac, r = result
            best_result = {
                "U": U_frac,
                "V": V_frac,
                "W": W_frac
            }
            best_rank = r
            break  # Found the lowest rank, stop
    
    # If no better rank found, return schoolbook
    if best_result is None:
        U = [[0] * rank_school for _ in range(dim_U)]
        V = [[0] * rank_school for _ in range(dim_V)]
        W = [[0] * rank_school for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
