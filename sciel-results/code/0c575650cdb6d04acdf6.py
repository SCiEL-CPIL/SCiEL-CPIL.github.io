"""Tensor decomposition search for 3x5x7 matrix multiplication."""
import time
import random
import math
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if U, V, W form a valid tensor decomposition.
    For each (i,j,k), the tensor entry T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] must equal 1.
    Using the convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    Index mapping: U[i*m+j], V[j*p+k], W[k*n+i].
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if s != 1:
                    return False
    return True

def _snap_to_half_int(val):
    """Snap a float to nearest half-integer."""
    r = round(val * 2) / 2
    return r

def _construct():
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with schoolbook
    best = _construct_schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank decomposition
    # For 3x5x7, the theoretical lower bound is around 25-30
    # Try ranks from 25 up to 104
    
    n, m, p = DIMENSIONS
    
    for target_rank in range(25, 105):
        if time.time() > deadline - 30:
            break
        
        # Randomized search for a valid decomposition
        # We'll use a simple approach: random initialization and gradient-free optimization
        # But this is very hard. Let's try a structured approach instead.
        
        # Try Kronecker product based approaches
        # 3x5x7: can we decompose 5 into 1+1+1+1+1 and use Strassen-like?
        
        # Actually, let's try a simpler heuristic:
        # Generate random sparse matrices with entries in {0, 1/2, 1}
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with random sparse patterns
            for c in range(target_rank):
                # Choose which (i,j,k) this column affects
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                val = random.choice([1, 1/2, -1/2])
                U[i * m + j][c] = val
                V[j * p + k][c] = val
                W[k * n + i][c] = val
            
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best

def construct():
    """Entry point: returns {'U', 'V', 'W'} for the tensor decomposition."""
    return _construct()
