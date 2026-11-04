"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import random

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = U[i*m + j][c]
                    v_val = V[j*p + k][c]
                    w_val = W[k*n + i][c]
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _try_rank(rank, n, m, p, max_attempts=200):
    """Try to find a rank decomposition of given rank using structured random search."""
    start_time = time.time()
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for attempt in range(max_attempts):
        if time.time() - start_time > 10:
            break
        
        # Strategy: Use structured random matrices with small integer entries
        # Initialize with random values from {-1, 0, 1, 2}
        U = [[0] * rank for _ in range(U_rows)]
        V = [[0] * rank for _ in range(V_rows)]
        W = [[0] * rank for _ in range(W_rows)]
        
        for i in range(U_rows):
            for c in range(rank):
                U[i][c] = random.randint(-2, 2)
        for i in range(V_rows):
            for c in range(rank):
                V[i][c] = random.randint(-2, 2)
        for i in range(W_rows):
            for c in range(rank):
                W[i][c] = random.randint(-2, 2)
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_rank_with_solving(rank, n, m, p, max_attempts=50):
    """Try to find a rank decomposition by fixing some variables and solving linear systems."""
    start_time = time.time()
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for attempt in range(max_attempts):
        if time.time() - start_time > 10:
            break
        
        # Fix U and V randomly, then solve for W
        # Or use a hybrid approach: fix most of U, V and solve for W
        
        # Randomly initialize U and V with small integers
        U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(U_rows)]
        V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(V_rows)]
        
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in W for fixed U, V
        # But it's not a simple linear system because W entries appear in multiple equations
        
        # Let's try a different approach: fix U, V, and solve for W column by column
        # Actually, this is complex. Let's just do random search with verification.
        
        W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(W_rows)]
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _strassen_block_decomposition(n, m, p):
    """
    Try to use Strassen-like block decomposition.
    For 2x3 * 3x4, we can pad to 2x4 * 4x4 or use other block structures.
    """
    # One approach: Pad 2x3 to 2x4 by adding a zero column
    # Then use 2x4 * 4x4 multiplication
    # But we need to find a good decomposition
    
    # Alternative: Use the fact that 2x3x4 can be embedded in 2x4x4
    # If we have a rank-r decomposition for 2x4x4, we can restrict it to 2x3x4
    
    # For now, let's try a simple block approach
    # Split the 3 columns into 1+2 or 2+1
    # This is complex, so let's just return None for now
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks starting from the theoretical lower bound
    # For 2x3x4, the lower bound is 19 (from various tensor rank bounds)
    # The schoolbook rank is 24
    
    # First try random search for low ranks
    for rank in range(19, 25):
        result = _try_rank(rank, n, m, p, max_attempts=100)
        if result is not None:
            U, V, W = result
            # Ensure entries are integers or half-integers
            U_out = [[float(u) for u in row] for row in U]
            V_out = [[float(v) for v in row] for row in V]
            W_out = [[float(w) for w in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # If random search fails, try the solving-based approach
    for rank in range(19, 25):
        result = _try_rank_with_solving(rank, n, m, p, max_attempts=30)
        if result is not None:
            U, V, W = result
            U_out = [[float(u) for u in row] for row in U]
            V_out = [[float(v) for v in row] for row in V]
            W_out = [[float(w) for w in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fall back to schoolbook decomposition of rank 24
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
# EVOLVE-BLOCK-END
