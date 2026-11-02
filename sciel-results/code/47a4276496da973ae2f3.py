"""
Program to find a low-rank exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
Uses a structured search over U,V with entries in {0,1} and solves for W.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _verify_decomposition(n, m, p, rank, U, V, W):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            s = Fraction(0)
            for k in range(p):
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
            if s != 1:
                return False
    return True

def _try_find_decomposition(n, m, p, rank, max_trials=100, timeout=None, start_time=0):
    """Try to find a rank-rank exact decomposition.
    Strategy: Fix U,V to random {0,1} matrices, solve for W using integer linear algebra."""
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = n * m * p
    
    # For each trial, random U and V with entries in {0, 1}
    for trial in range(max_trials):
        if timeout is not None and time.time() - start_time > timeout:
            return None
        
        # Random U and V
        U = [[random.choice([0, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([0, 1]) for _ in range(rank)] for _ in range(mp)]
        
        # Build the system A w = b where:
        # For each (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
        # So A is (num_eqs) x (rank * pn), and b is ones.
        # A[e, r*pn + k*n + i] = U[i*m+j][r] * V[j*p+k][r]
        
        # Instead of building the full sparse system, let's use a smarter approach.
        # For fixed U, V, we need to find W such that the equation holds.
        # This is a system of linear equations in the entries of W.
        
        # Let's build A as a numpy matrix
        A = np.zeros((num_eqs, rank * pn), dtype=float)
        b = np.ones(num_eqs, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = i * m * p + j * p + k
                    for r in range(rank):
                        u = r * pn + k * n + i
                        A[e, u] = float(U[i*m+j][r] * V[j*p+k][r])
        
        # Solve using least squares
        try:
            w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is close to integers or half-integers
        W_flat = w.reshape(rank, pn)
        
        # Check if entries are close to integers
        all_int = True
        for r in range(rank):
            for idx in range(pn):
                val = W_flat[r, idx]
                nearest_int = round(val)
                if abs(val - nearest_int) > 1e-6:
                    all_int = False
                    break
            if not all_int:
                break
        
        if all_int:
            # Construct integer W
            W = [[0] * rank for _ in range(pn)]
            for r in range(rank):
                for idx in range(pn):
                    W[idx][r] = int(round(W_flat[r, idx]))
            
            # Verify exactness
            if _verify_decomposition(n, m, p, rank, U, V, W):
                return U, V, W, rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Known lower bound: rank >= max(nm, mp, pn) = max(6, 18, 12) = 18
    # But we know rank(2x3x6) might be lower. Let's try from 12 upwards.
    # Actually, the trivial schoolbook is 36. Let's search from 12 to 35.
    
    best_result = None
    
    # Try ranks from 12 to 35
    for rank in range(12, 36):
        if time.time() - start_time > deadline:
            break
        
        # Number of trials depends on rank - smaller ranks may need more trials
        # but the system is harder. Let's use a budget.
        remaining = deadline - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Allocate trials based on remaining time
        # Each trial takes roughly O(num_eqs * rank * pn) for building A
        # num_eqs = 36, rank*pn up to 35*12 = 420, so A is 36x420
        # lstsq on 36x420 is fast, maybe 1ms
        # Let's do as many trials as we can in the remaining time, with a cap.
        
        max_trials = min(500, max(10, int(remaining / 0.01)))
        
        result = _try_find_decomposition(n, m, p, rank, max_trials, timeout=remaining, start_time=start_time)
        if result is not None:
            U, V, W, r = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook rank 36
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
