"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks and standard multiplication for 2x2x1 blocks.
    Rank = 50.
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Shapes
    u_size = n * m  # 8
    v_size = m * p  # 28
    w_size = p * n  # 14
    
    rank = 50
    U = [[0] * rank for _ in range(u_size)]
    V = [[0] * rank for _ in range(v_size)]
    W = [[0] * rank for _ in range(w_size)]
    
    # Define Strassen factors for 2x2x2
    # Local indices: a (0,1), b (0,1), c (0,1)
    # U_idx = a*2 + b
    # V_idx = b*2 + c
    # W_idx = c*2 + a
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: A00(-1), A10(1)
        [0, 1, 0, -1] # M7: A01(1), A11(-1)
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01(1), B11(-1)
        [-1, 0, 1, 0],# M4: B00(-1), B10(1)
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: C00(1), C11(1) -> W_idx: c*2+a. C00 (c=0,a=0)->0, C11(c=1,a=1)->3
        [0, 1, 0, 0], # M2: C10(1) -> W_idx: c=0, a=1 -> 1
        [0, 0, 1, 0], # M3: C01(1) -> W_idx: c=1, a=0 -> 2
        [1, 1, 0, 0], # M4: C00(1), C10(1) -> W_idx: 0, 1
        [-1, 0, 1, 0],# M5: C00(-1), C01(1) -> W_idx: 0(-1), 2(1)
        [0, 0, 0, 1], # M6: C11(1) -> W_idx: 3
        [1, 0, 0, 1]  # M7: C00(1), C11(1) -> W_idx: 0, 3
    ]
    
    # Define Standard factors for 2x2x1
    # Local indices: a (0,1), b (0,1), c (0)
    # U_idx = a*2 + b
    # V_idx = b*1 + 0 = b
    # W_idx = c*2 + a = a
    
    std_U = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    
    std_V = [
        [1, 0, 1, 0], # B00 (for A00, A10)
        [0, 1, 0, 1]  # B10 (for A01, A11)
    ]
    
    std_W = [
        [1, 1, 0, 0], # C0 (from A00, A01)
        [0, 0, 1, 1]  # C1 (from A10, A11)
    ]
    
    # Assemble
    # q: 0, 1 (block of A/B rows)
    # s: 0, 1, 2 (Strassen 2x2x2), 3 (Standard 2x2x1)
    
    for q in range(2):
        j_base = 2 * q
        
        for s in range(3):
            k_base = 2 * s
            rank_base = 7 * (2 * q + s)
            
            for t in range(7):
                r = rank_base + t
                
                # Fill U
                for a in range(2):
                    for b in range(2):
                        val = strassen_U[t][a * 2 + b]
                        if val != 0:
                            u_idx = a * 4 + (j_base + b)
                            U[u_idx][r] += val
                            
                # Fill V
                for b in range(2):
                    for c in range(2):
                        val = strassen_V[t][b * 2 + c]
                        if val != 0:
                            v_idx = (j_base + b) * 7 + (k_base + c)
                            V[v_idx][r] += val
                            
                # Fill W
                for c in range(2):
                    for a in range(2):
                        val = strassen_W[t][c * 2 + a]
                        if val != 0:
                            w_idx = (k_base + c) * 2 + a
                            W[w_idx][r] += val
                            
        # Standard 2x2x1 block
        k_base = 6
        rank_base = 7 * (2 * q + 3)
        
        for t in range(4):
            r = rank_base + t
            
            # Fill U
            for a in range(2):
                for b in range(2):
                    val = std_U[t][a * 2 + b]
                    if val != 0:
                        u_idx = a * 4 + (j_base + b)
                        U[u_idx][r] += val
                        
            # Fill V
            for b in range(2):
                val = std_V[t][b]
                if val != 0:
                    v_idx = (j_base + b) * 7 + k_base
                    V[v_idx][r] += val
                    
            # Fill W
            for a in range(2):
                val = std_W[t][a]
                if val != 0:
                    w_idx = k_base * 2 + a
                    W[w_idx][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
