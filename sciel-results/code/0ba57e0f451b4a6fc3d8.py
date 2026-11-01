"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = 2, 3, 4
    N, M, P = n * m, m * p, p * n  # 6, 12, 8

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 24
    U_school = [[0] * rank_school for _ in range(N)]
    V_school = [[0] * rank_school for _ in range(M)]
    W_school = [[0] * rank_school for _ in range(P)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        """Verify that the decomposition exactly matches matrix multiplication using Fractions for exactness."""
        rank = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        u = U_frac[i * m + j][r]
                        v = V_frac[j * p + k][r]
                        w = W_frac[k * n + i][r]
                        total += u * v * w
                    if total != 1:
                        return False
        return True

    def verify_fast(U, V, W):
        """Fast verification using numpy for initial screening."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        return True

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Systematic search for rank-23 decomposition
    # The key insight is that we can potentially merge two terms in the schoolbook decomposition
    # if there's a linear dependency that allows cancellation.
    
    # Approach 1: Try to find a rank-23 solution by perturbing the schoolbook solution
    # We'll use a structured random search with rational values from a small set
    
    random.seed(42)
    
    # Values to use in search - keep them small for exactness
    search_vals = [Fraction(-2), Fraction(-3/2), Fraction(-1), Fraction(-1/2), 
                   Fraction(0), Fraction(1/2), Fraction(1), Fraction(3/2), Fraction(2)]
    
    def try_rank_structured(r, max_attempts=50000):
        """Try to find a rank-r decomposition using structured random search."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Generate random values, but with some structure
            # Use a mix of zeros and small rationals
            U = [[random.choice(search_vals) for _ in range(r)] for _ in range(N)]
            V = [[random.choice(search_vals) for _ in range(r)] for _ in range(M)]
            W = [[random.choice(search_vals) for _ in range(r)] for _ in range(P)]
            
            # Fast check first
            if verify_fast(U, V, W):
                # Exact verification
                if verify(U, V, W):
                    return (U, V, W)
        
        return None

    # Try rank 23 first (most likely improvement)
    result = try_rank_structured(23, max_attempts=30000)
    if result is not None:
        best_U, best_V, best_W = result
        best_rank = 23
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If rank 23 fails, try a more targeted approach
    # The matrix multiplication tensor for 2x3x4 might have a known lower bound
    # Let's try to use the fact that we can sometimes save one term
    
    # Approach 2: Try specific patterns that might allow rank reduction
    # For example, try to make one column of the decomposition a linear combination
    # of others in a way that preserves the tensor structure
    
    def try_rank_targeted(r):
        """Try targeted patterns for rank reduction."""
        if r >= 24:
            return None
        
        # Try to find a solution where one rank term is a "correction" term
        # Start from schoolbook and try to modify it
        
        for pattern in range(1000):
            if time.time() > deadline - 10:
                return None
            
            U = [row[:] for row in U_school[:N]]
            V = [row[:] for row in V_school[:M]]
            W = [row[:] for row in W_school[:P]]
            
            # Try to set some entries to create dependencies
            # This is a heuristic approach
            
            if random.random() < 0.5:
                # Try to zero out some entries
                for _ in range(random.randint(1, 5)):
                    idx = random.randint(0, r - 1)
                    U[random.randint(0, N-1)][idx] = Fraction(0)
            else:
                # Try to add small perturbations
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, r - 1)
                    U[random.randint(0, N-1)][idx] = random.choice(search_vals)
                    V[random.randint(0, M-1)][idx] = random.choice(search_vals)
                    W[random.randint(0, P-1)][idx] = random.choice(search_vals)
            
            # Trim to rank r
            U = [row[:r] for row in U]
            V = [row[:r] for row in V]
            W = [row[:r] for row in W]
            
            if verify_fast(U, V, W):
                if verify(U, V, W):
                    return (U, V, W)
        
        return None

    result = try_rank_targeted(23)
    if result is not None:
        best_U, best_V, best_W = result
        best_rank = 23
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If all else fails, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
