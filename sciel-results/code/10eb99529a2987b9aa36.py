"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of invalid entries to ensure they are 0
    # An entry (a, b, c) is invalid if the indices don't form a consistent (i, j, k)
    # For n=2, m=2, p=16, total entries 4*32*32 = 4096. We can check all.
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val) > tol:
                        return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen steps for 2x2x2
    # u_local: index 0: a00, 1: a01, 2: a10, 3: a11
    # v_local: index 0: b00, 1: b01, 2: b10, 3: b11
    # w_local: index 0: c00, 1: c01, 2: c10, 3: c11
    
    u_steps = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    v_steps = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_steps = [
        [1, 0, 0, 1], # M1 -> c00, c11
        [0, 0, 1, -1], # M2 -> c10, c11
        [0, 1, 0, 1], # M3 -> c01, c11
        [1, 0, 1, 0], # M4 -> c00, c10
        [-1, 1, 0, 0], # M5 -> c00, c01
        [0, 0, 0, 1], # M6 -> c11
        [1, 0, 0, 0]  # M7 -> c00
    ]
    
    num_blocks = p // 2 # 8 blocks
    
    for block in range(num_blocks):
        # Global indices for V (B matrix)
        # B is 2x16. v_idx = j * 16 + k
        # Block uses k = 2*block, 2*block+1
        # b00: j=0, k=2b -> 2b
        # b01: j=0, k=2b+1 -> 2b+1
        # b10: j=1, k=2b -> 16+2b
        # b11: j=1, k=2b+1 -> 16+2b+1
        v_map = [2 * block, 2 * block + 1, 16 + 2 * block, 16 + 2 * block + 1]
        
        # Global indices for W (C matrix)
        # C is 2x16. w_idx = k * 2 + i
        # c00: i=0, k=2b -> 4b
        # c01: i=0, k=2b+1 -> 4b+2
        # c10: i=1, k=2b -> 4b+1
        # c11: i=1, k=2b+1 -> 4b+3
        w_map = [4 * block, 4 * block + 2, 4 * block + 1, 4 * block + 3]
        
        # Global indices for U (A matrix)
        # A is 2x2. u_idx = i * 2 + j
        # a00: 0, a01: 1, a10: 2, a11: 3
        u_map = [0, 1, 2, 3]
        
        for s in range(7):
            r = block * 7 + s
            
            # Fill U
            for l in range(4):
                if u_steps[s][l] != 0:
                    U[u_map[l], r] = u_steps[s][l]
            
            # Fill V
            for l in range(4):
                if v_steps[s][l] != 0:
                    V[v_map[l], r] = v_steps[s][l]
            
            # Fill W
            for l in range(4):
                if w_steps[s][l] != 0:
                    W[w_map[l], r] = w_steps[s][l]
                    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
