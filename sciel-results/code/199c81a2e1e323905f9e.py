"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = len(U_np[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c2 = k * n + i
                val = sum(U_np[a, c] * V_np[b, c] * W_np[c2, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    # Also check that off-diagonal terms are zero
    # T[a,b,c2] should be 0 when indices are inconsistent
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 != j:
                        a = i * m + j
                        b = j2 * p + k
                        c2 = k * n + i
                        val = sum(U_np[a, c] * V_np[b, c] * W_np[c2, c] for c in range(rank))
                        if abs(val) > tol:
                            return False
                for i2 in range(n):
                    if i2 != i:
                        a = i * m + j
                        b = j * p + k
                        c2 = k * n + i2
                        val = sum(U_np[a, c] * V_np[b, c] * W_np[c2, c] for c in range(rank))
                        if abs(val) > tol:
                            return False
                for k2 in range(p):
                    if k2 != k:
                        a = i * m + j
                        b = j * p + k2
                        c2 = k2 * n + i
                        val = sum(U_np[a, c] * V_np[b, c] * W_np[c2, c] for c in range(rank))
                        if abs(val) > tol:
                            return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Try to find lower rank by random search with integer/half-integer entries
    # The tensor has structure: T[i,j,k] = 1 iff indices consistent, else 0
    
    # For 2x2x9, let's try ranks from 35 down to some minimum
    # Use random initialization and local search
    
    random.seed(42)
    
    # Try a structured approach first:
    # Since m=2, we can think of this as 2x2 matrices times 2x9 matrices
    # The key is that for each (i,k) output, we compute A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Let's try to use a "Strassen-like" approach by sharing multiplications
    # across different k values
    
    # Approach: Random search with constraint satisfaction
    # Try ranks from best_rank-1 down to max(18, best_rank-10)
    
    for target_rank in range(best_rank - 1, max(17, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random integer/half-integer values
            # Use small values to keep things manageable
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            for c in range(target_rank):
                for a in range(n * m):
                    U[a, c] = random.choice(vals)
                for b in range(m * p):
                    V[b, c] = random.choice(vals)
                for c2 in range(p * n):
                    W[c2, c] = random.choice(vals)
            
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # If no improvement found, return schoolbook
    return best
# EVOLVE-BLOCK-END
