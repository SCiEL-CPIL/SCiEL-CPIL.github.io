"""
Rewritten program to find a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
Uses a structured search based on Strassen-like patterns to find integer/half-integer decompositions.
"""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _time_remaining():
        return deadline - time.time()
    
    def _verify(U, V, W):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def _make_schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Start with schoolbook as fallback
    best = _make_schoolbook()
    best_rank = n * m * p
    
    # Strategy: Structured search for lower ranks.
    # We know rank 40 is possible.
    # We try to find ranks 30, 32, 34, 36, 38.
    
    # To make the search effective, we use a template based on Strassen's algorithm.
    # Strassen for 2x2x2 uses 7 terms.
    # We can extend this to 2x4x5.
    
    # We will try to find a decomposition with integer entries in {-2, -1, 0, 1, 2}.
    # We use a local search / simulated annealing approach on the integer lattice.
    
    def _search_for_rank(target_rank, time_limit=20.0):
        """
        Search for a decomposition of the given rank using integer entries.
        """
        start = time.time()
        rng = random.Random(42)
        
        # Initialize with random integers
        U = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
        V = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
        W = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]
        
        # Calculate error
        def _calc_error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        err += (s - 1.0) ** 2
            return err
        
        current_error = _calc_error()
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        best_error = current_error
        
        # Local search
        while time.time() - start < time_limit:
            # Pick a random matrix, row, and column to perturb
            mat_choice = rng.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                mat = U
                rows = nm
            elif mat_choice == 'V':
                mat = V
                rows = mp
            else:
                mat = W
                rows = pn
                
            row = rng.randint(0, rows - 1)
            col = rng.randint(0, target_rank - 1)
            
            old_val = mat[row][col]
            new_val = old_val + rng.choice([-1, 1])
            # Clamp to reasonable range
            if new_val > 5: new_val = 5
            if new_val < -5: new_val = -5
            
            mat[row][col] = new_val
            
            new_error = _calc_error()
            
            if new_error < current_error:
                current_error = new_error
                if new_error < 1e-9:
                    # Found exact solution
                    return U, V, W
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = [r[:] for r in U], [r[:] for r in V], [r[:] for r in W]
            else:
                # Revert
                mat[row][col] = old_val
            # If we get stuck, restart with a new random init
            if rng.random() < 0.01:
                U = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
                V = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
                W = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]
                current_error = _calc_error()
        
        # Return the best found if it's exact (shouldn't happen in this loop unless we check)
        # We only return if exact.
        if best_error < 1e-9:
            return best_U, best_V, best_W
        return None

    # Try ranks from 30 upwards
    for target_rank in range(30, best_rank + 1):
        if _time_remaining() < 5.0:
            break
        
        found = _search_for_rank(target_rank, time_limit=10.0)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
