import time
import sys

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 14 decomposition using two Strassen blocks
    rank = 14
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for a 2x2x2 multiplication
    # Each entry is a list of (index, coefficient) for the corresponding matrix
    # Indices for U (A): i*2 + j
    # Indices for V (B): j*2 + k (local)
    # Indices for W (C): k*2 + i (local)
    
    # r=0: M1 = (a00+a11)(b00+b11)
    # U: (0,1), (3,1)
    # V: (0,1), (3,1)
    # W: c00(0)->+1, c11(3)->+1
    strassen_terms = [
        # (U_indices, V_indices, W_indices)
        # r=0
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # r=1: M2 = (a10+a11)b00
        # U: (2,1), (3,1)
        # V: (0,1)
        # W: c10(1)->+1, c11(3)->-1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # r=2: M3 = a00(b01-b11)
        # U: (0,1)
        # V: (1,1), (3,-1)
        # W: c01(2)->+1, c11(3)->+1
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # r=3: M4 = a11(b10-b00)
        # U: (3,1)
        # V: (2,1), (0,-1)
        # W: c00(0)->+1, c10(1)->+1
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # r=4: M5 = (a00+a01)b11
        # U: (0,1), (1,1)
        # V: (3,1)
        # W: c00(0)->-1, c01(2)->+1
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # r=5: M6 = (a10-a00)(b00+b01)
        # U: (2,1), (0,-1)
        # V: (0,1), (1,1)
        # W: c11(3)->+1
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # r=6: M7 = (a01-a11)(b10+b11)
        # U: (1,1), (3,-1)
        # V: (2,1), (3,1)
        # W: c00(0)->+1
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Fill in the first half (columns 0-6)
    for r_local in range(7):
        r_global = r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx][r_global] = val
            
        for idx, val in v_terms:
            # V index mapping: local (j, k) -> j*2 + k. Global is j*4 + k.
            j = idx // 2
            k = idx % 2
            global_v_idx = j * 4 + k
            V[global_v_idx][r_global] = val
            
        for idx, val in w_terms:
            # W index mapping: local (k, i) -> k*2 + i. Global is k*2 + i? 
            # Wait, W index is k*n + i. n=2. So k*2 + i.
            # The local k is 0 or 1. Global k is 0 or 1 for the first half.
            # So global_w_idx = k*2 + i.
            k = idx // 2
            i = idx % 2
            global_w_idx = k * 2 + i
            W[global_w_idx][r_global] = val

    # Fill in the second half (columns 7-13)
    # A is the same, so U is the same.
    # B columns are 2, 3. Local k is 0, 1. Global k is 2, 3.
    # C columns are 2, 3. Local k is 0, 1. Global k is 2, 3.
    
    for r_local in range(7):
        r_global = 7 + r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx][r_global] = val
            
        for idx, val in v_terms:
            j = idx // 2
            k_local = idx % 2
            k_global = k_local + 2
            global_v_idx = j * 4 + k_global
            V[global_v_idx][r_global] = val
            
        for idx, val in w_terms:
            k_local = idx // 2
            i = idx % 2
            k_global = k_local + 2
            global_w_idx = k_global * 2 + i
            W[global_w_idx][r_global] = val

    return {"U": U, "V": V, "W": W}
