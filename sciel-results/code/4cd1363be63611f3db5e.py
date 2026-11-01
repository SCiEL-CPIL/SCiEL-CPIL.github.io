import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def calculate_defects(U, V, W, n, m, p):
    """Calculate the difference between the decomposed tensor and the target tensor (all 1s)."""
    r = len(W[0]) if W else 0
    defects = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                defects.append(val - 1.0)
    return defects

def integer_local_search(n, m, p, rank, deadline, max_iter=5000):
    """
    Perform local search on integer/half-integer grid.
    """
    # Initialize with small random integers
    U = np.random.randint(-2, 3, size=(n * m, rank))
    V = np.random.randint(-2, 3, size=(m * p, rank))
    W = np.random.randint(-2, 3, size=(p * n, rank))
    
    # Convert to lists for easier modification if needed, but numpy is faster for defect calc
    # We will use numpy arrays for speed in defect calculation
    
    current_defects = calculate_defects(U.tolist(), V.tolist(), W.tolist(), n, m, p)
    current_error = np.sum(np.square(current_defects))
    
    if current_error < 1e-9:
        return U.tolist(), V.tolist(), W.tolist()

    step_options = [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        # Pick a random matrix and entry
        mat_idx = np.random.randint(0, 3)
        if mat_idx == 0:
            mat = U
            rows = n * m
        elif mat_idx == 1:
            mat = V
            rows = m * p
        else:
            mat = W
            rows = p * n
            
        row = np.random.randint(0, rows)
        col = np.random.randint(0, rank)
        
        original_val = mat[row, col]
        
        # Try a random step
        step = random.choice(step_options)
        new_val = original_val + step
        
        # Ensure we stay in reasonable bounds to avoid overflow or huge errors
        if abs(new_val) > 5:
            continue
            
        # Calculate change in error
        # Delta error = sum over all tensor entries (i,j,k) involving this term
        # This is expensive if done naively. 
        # However, for small ranks and dimensions, it's manageable.
        # A term U[row, col] appears in all (i,j,k) where i*m+j == row.
        # We can precompute which tensor indices are affected.
        
        # For simplicity and speed in this specific small case (50 tensor entries),
        # we just recalculate the full error if the move is promising, 
        # or use a cheaper heuristic.
        
        # Let's try a simple hill climbing:
        # Temporarily apply change
        mat[row, col] = new_val
        
        new_defects = calculate_defects(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        new_error = np.sum(np.square(new_defects))
        
        if new_error < current_error:
            current_error = new_error
            current_defects = new_defects
            if current_error < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
        else:
            # Revert
            mat[row, col] = original_val
            
    return None

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Start with schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    best_result = (U, V, W)
    best_rank = len(W[0])
    
    # Try to find lower ranks
    # Prioritize ranks that are likely to exist based on known bounds
    # For 2x5x5, schoolbook is 50.
    # We can try 49, 48, ... down to 40.
    # Integer local search is slower per iteration but more precise.
    
    candidate_ranks = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break
            
        # Try multiple random starts for local search
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            result = integer_local_search(n, m, p, rank, deadline, max_iter=2000)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a lower rank, continue to try even lower
            pass

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
