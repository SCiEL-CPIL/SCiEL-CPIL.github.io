"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, provide the schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=object)
        V = np.zeros((m * p, rank), dtype=object)
        W = np.zeros((p * n, rank), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if total != 1:
                        return False
        return True
    
    # Convert to proper types for output
    def to_output(U, V, W, rank):
        # Convert Fraction to int or half-integer representation
        # The problem says "integer/half-integer entries"
        # We'll keep them as fractions for exactness, but the evaluator should handle this
        # Actually, let's return them as floats that are exactly representable
        U_out = np.zeros((n * m, rank))
        V_out = np.zeros((m * p, rank))
        W_out = np.zeros((p * n, rank))
        for i in range(n * m):
            for c in range(rank):
                U_out[i, c] = float(U[i, c])
        for i in range(m * p):
            for c in range(rank):
                V_out[i, c] = float(V[i, c])
        for i in range(p * n):
            for c in range(rank):
                W_out[i, c] = float(W[i, c])
        return U_out, V_out, W_out
    
    # Try to find a lower rank decomposition
    # Strategy: Random search with small integer/half-integer values
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)
    
    for target_rank in range(lower_bound, n * m * p):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values from {-1, -0.5, 0, 0.5, 1}
            values = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]
            
            U = np.zeros((n * m, target_rank), dtype=object)
            V = np.zeros((m * p, target_rank), dtype=object)
            W = np.zeros((p * n, target_rank), dtype=object)
            
            for i in range(n * m):
                for c in range(target_rank):
                    U[i, c] = random.choice(values)
            for i in range(m * p):
                for c in range(target_rank):
                    V[i, c] = random.choice(values)
            for i in range(p * n):
                for c in range(target_rank):
                    W[i, c] = random.choice(values)
            
            if verify(U, V, W, target_rank):
                found = True
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if found:
            break
    
    U_out, V_out, W_out = to_output(best_U, best_V, best_W, best_rank)
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
