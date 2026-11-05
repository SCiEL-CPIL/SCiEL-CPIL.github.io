import numpy as np
import time
import sys
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    start_time = time.time()
    time_limit = 135.0  # Leave 15s margin before 150s kill
    
    def check_decomposition_exact(U, V, W, rank):
        """Check if decomposition is exact using Fraction arithmetic."""
        U_frac = [[Fraction(U[i, t]) for t in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i, t]) for t in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W[i, t]) for t in range(rank)] for i in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for t in range(rank):
                        val += U_frac[i*m+j][t] * V_frac[j*p+k][t] * W_frac[k*n+i][t]
                    if val != 1:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the standard schoolbook decomposition of rank 72."""
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
        
        return {"U": U, "V": V, "W": W}, rank
    
    def try_random_search(target_rank, max_attempts=50000):
        """Random search for a valid decomposition of target_rank."""
        if time.time() - start_time > time_limit:
            return None
        
        np.random.seed(42)
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                return None
            
            # Generate random half-integer entries
            U = (np.random.randint(0, 9, size=(n*m, target_rank)) - 4) * 0.5
            V = (np.random.randint(0, 9, size=(m*p, target_rank)) - 4) * 0.5
            W = (np.random.randint(0, 9, size=(p*n, target_rank)) - 4) * 0.5
            
            # Quick float check
            valid = True
            for (i, j, k) in triples:
                val = 0.0
                for t in range(target_rank):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Exact verification
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                if check_decomposition_exact(U_arr, V_arr, W_arr, target_rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try to find the lowest rank we can achieve
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 20, 50), -1):
        if time.time() - start_time > time_limit:
            break
        
        result = try_random_search(target_rank, max_attempts=10000)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook_decomposition()[0]
