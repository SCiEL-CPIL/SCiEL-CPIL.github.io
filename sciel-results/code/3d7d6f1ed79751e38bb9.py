"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """Return the lowest-rank tensor decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    1. Try to find a rank-11 decomposition for 2x3x3 (known to exist)
    2. Use block construction to get rank 33 for 2x3x9
    3. Fallback to higher ranks or schoolbook (54)
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
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
    
    def verify_233(U, V, W, R):
        """Verify 2x3x3 decomposition exactly.
        U: (6, R), V: (9, R), W: (6, R)
        i in 0..1, j in 0..2, k in 0..2
        U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
        """
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(R):
                        s += U[i*3+j][c] * V[j*3+k][c] * W[k*2+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_239(U, V, W, R):
        """Verify 2x3x9 decomposition exactly.
        U: (6, R), V: (27, R), W: (18, R)
        i in 0..1, j in 0..2, k in 0..8
        U[i*3+j, c], V[j*9+k, c], W[k*2+i, c]
        """
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    s = 0
                    for c in range(R):
                        s += U[i*3+j][c] * V[j*9+k][c] * W[k*2+i][c]
                    if s != 1:
                        return False
        return True
    
    def build_239_from_233(decomp_233):
        """Build 2x3x9 decomposition from 2x3x3 decomposition."""
        R = len(decomp_233["U"][0])
        total_rank = 3 * R
        U = [[0] * total_rank for _ in range(6)]
        V = [[0] * total_rank for _ in range(27)]
        W = [[0] * total_rank for _ in range(18)]
        
        U_233 = decomp_233["U"]
        V_233 = decomp_233["V"]
        W_233 = decomp_233["W"]
        
        for block in range(3):
            for c in range(R):
                col = block * R + c
                for row in range(6):
                    U[row][col] = U_233[row][c]
                for j in range(3):
                    for k_local in range(3):
                        k_actual = 3 * block + k_local
                        V[j * 9 + k_actual][col] = V_233[j * 3 + k_local][c]
                for k_local in range(3):
                    for i in range(2):
                        k_actual = 3 * block + k_local
                        W[k_actual * 2 + i][col] = W_233[k_local * 2 + i][c]
        
        return {"U": U, "V": V, "W": W}
    
    def find_233_decomp_structured(target_rank, max_trials=5000):
        """Search for 2x3x3 decomposition with structured random entries."""
        # Use entries from a small set to increase chance of exact solutions
        entries = [-2, -1, 0, 1, 2]
        
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Generate sparse random matrices with structured entries
            density = 0.25 + 0.1 * (target_rank - 10)  # slightly denser for higher ranks
            
            U = [[random.choice(entries) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(6)]
            V = [[random.choice(entries) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(9)]
            W = [[random.choice(entries) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(6)]
            
            if verify_233(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def find_233_decomp_binary(target_rank, max_trials=5000):
        """Search with binary entries {-1, 0, 1}."""
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE:
                return None
            
            density = 0.3
            
            U = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(6)]
            V = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(9)]
            W = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(6)]
            
            if verify_233(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find a low-rank 2x3x3 decomposition
    # Known: rank(2x3x3) = 11
    best_233 = None
    
    # Try rank 11 first (known to be achievable)
    for R in [11, 12, 13, 14]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Use both search strategies
        decomp = find_233_decomp_binary(R, 3000)
        if decomp:
            best_233 = decomp
            break
        
        if time.time() - start_time > DEADLINE:
            break
        
        decomp = find_233_decomp_structured(R, 2000)
        if decomp:
            best_233 = decomp
            break
    
    if best_233:
        R_233 = len(best_233["U"][0])
        result = build_239_from_233(best_233)
        # Verify the 2x3x9 result
        R_239 = len(result["U"][0])
        if verify_239(result["U"], result["V"], result["W"], R_239):
            return result
        # If verification fails (shouldn't happen), fall through
    
    # Fallback: try direct search for 2x3x9 with rank < 54
    # This is unlikely to succeed quickly, but try a bit
    if time.time() - start_time < DEADLINE - 10:
        # Try rank 36 (3 * 12) as a moderate target
        target_rank = 36
        for trial in range(500):
            if time.time() - start_time > DEADLINE - 5:
                break
            density = 0.2
            U = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(6)]
            V = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(27)]
            W = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(target_rank)] for _ in range(18)]
            
            if verify_239(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    # Ultimate fallback: schoolbook
    return build_schoolbook()
