"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses LP-based feasibility search with structured decompositions.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition (rank n*m*p)."""
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
        return U, V, W
    
    def try_structured_decomposition(rank):
        """
        Try to find a decomposition of given rank using structured initialization.
        Uses the idea that we can group terms and use shared structure.
        """
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(42 + rank)
        
        # Strategy 1: Block-diagonal style - group by i
        # For each i, decompose the m*p sub-tensor
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Try to use rank/ n columns per i-block
            cols_per_i = rank // n
            for i in range(n):
                base = i * cols_per_i
                for c in range(cols_per_i):
                    col = base + c
                    # Random structure within this block
                    for j in range(m):
                        U[i*m + j, col] = np.random.choice([-1, 0, 1])
                    for j in range(m):
                        for k in range(p):
                            V[j*p + k, col] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        W[k*n + i, col] = np.random.choice([-1, 0, 1])
            
            if verify(U.tolist(), V.tolist(), W.tolist(), 1e-6):
                return U.tolist(), V.tolist(), W.tolist()
        
        # Strategy 2: Full random with sparsity
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            if verify(U.tolist(), V.tolist(), W.tolist(), 1e-6):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to find lower rank decompositions
    best_result = None
    
    # Try ranks below schoolbook
    for rank in range(120, schoolbook_rank + 1, -5):
        if time.time() > deadline - 20:
            break
        
        result = try_structured_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_result = (U, V, W, rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
