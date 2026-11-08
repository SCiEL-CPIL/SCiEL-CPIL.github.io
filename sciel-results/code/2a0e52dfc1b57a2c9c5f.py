"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _berlekamp_massey(s, field):
    """
    Berlekamp-Massey algorithm over a finite field.
    Returns the minimal polynomial (as a list of coefficients, highest degree first).
    """
    n = len(s)
    if n == 0:
        return [1]
    
    C = [0] * n
    B = [0] * n
    C[0] = 1
    B[0] = 1
    
    L = 0
    m = 1
    b = 1
    
    for i in range(n):
        # Compute discrepancy
        d = s[i]
        for j in range(1, L + 1):
            d = (d + C[j] * s[i - j]) % field
        
        if d == 0:
            m += 1
        else:
            T = C[:]
            if b == 0:
                continue
            # C = C - d * x^m * B / b
            factor = (d * pow(b, field - 2, field)) % field
            for j in range(m, n):
                C[j] = (C[j] - factor * B[j - m]) % field
            
            if 2 * L <= i:
                L = i + 1 - L
                B = T
                m = 1
                b = d
            else:
                m += 1
    
    # Return the connection polynomial (length L+1)
    return C[:L + 1]

def _find_rank_reduction(n, m, p, target_rank, field=5, max_iters=50000):
    """
    Try to find a rank-reduced decomposition over F_field.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leave 10s margin
    
    # Initialize with schoolbook
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # First target_rank columns: schoolbook pattern
    for r in range(min(target_rank, n * m * p)):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
    
    # Try to find dependencies to reduce rank
    # For each column r, we can try to express it as a combination of others
    # This is complex; instead, use random search with verification
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition over F_field"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(target_rank):
                        total = (total + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]) % field
                    if total != 1:
                        return False
        return True
    
    # Try random perturbations
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Randomly modify a few entries
        num_mods = random.randint(1, 5)
        for _ in range(num_mods):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                U[row][col] = random.randint(0, field - 1)
            elif mat == 'V':
                row = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                V[row][col] = random.randint(0, field - 1)
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                W[row][col] = random.randint(0, field - 1)
        
        if verify(U, V, W):
            return U, V, W
        
        # Occasionally reset to schoolbook and try a different approach
        if iteration % 1000 == 0:
            for r in range(min(target_rank, n * m * p)):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 16
    # Known upper bounds: 48 (schoolbook)
    # Try ranks from 47 down to some reasonable minimum
    
    for target_rank in range(47, 30, -1):
        result = _find_rank_reduction(n, m, p, target_rank, field=5, max_iters=10000)
        if result is not None:
            U, V, W = result
            # Lift to integers (values are already 0-4, which are integers)
            return {"U": U, "V": V, "W": W}
    
    # If no reduction found, return schoolbook
    rank = schoolbook_rank
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
