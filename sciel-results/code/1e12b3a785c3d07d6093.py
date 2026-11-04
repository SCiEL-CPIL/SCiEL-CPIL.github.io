"""Rewritten program using Strassen-like decomposition for rank reduction."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # 1. A12 (2x2) * B12 (2x10) using 5 Strassen blocks (rank 35)
    # 2. A3 (2x1) * B3 (1x10) using outer product (rank 20)
    
    R = 55
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    r_idx = 0
    
    # Part 1: Strassen for A12 * B12
    # A12 indices: i in 0..1, j in 0..1. Flat index i*3 + j
    # B12 indices: j in 0..1, k in 0..9. Flat index j*10 + k
    # C indices: k in 0..9, i in 0..1. Flat index k*2 + i
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B00 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: list of (i, j, coeff)
        # B_coeffs: list of (j, k, coeff)
        # C_coeffs: list of (k, i, coeff)
        # Note: k is relative to block start
        
        # M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1), (0,1,-1), (1,0,1) ]), # C00+=1, C11+=1, C01? No.
        # Let's map C indices carefully.
        # C00 (k=0, i=0), C01 (k=1, i=0), C10 (k=0, i=1), C11 (k=1, i=1)
        # M1 contributes to C00 and C11.
        # C00: +1. C11: +1.
        # So C_coeffs: (0,0,1), (1,1,1).
        
        # M2
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (0,1,-1), (1,0,1) ]), # C01? No.
        # M2 = (A10+A11)B00.
        # C10 = M2 + M4. C11 = M1 - M2 + ...
        # So M2 contributes to C10 (+1) and C11 (-1).
        # C10 is k=0, i=1. C11 is k=1, i=1.
        
        # M3
        ([ (0,0,1) ], [ (1,0,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]), # C01 + M3, C11 + M3
        
        # M4
        ([ (1,1,1) ], [ (0,0,1), (1,0,-1) ], [ (0,0,1), (1,0,1) ]), # C00 + M4, C10 + M4
        
        # M5
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]), # C00 - M5, C01 + M5
        
        # M6
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]), # C11 + M6
        
        # M7
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]), # C00 + M7
    ]
    
    # Correcting C_coeffs based on standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # M2
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]), # C10(k=0,i=1), C11(k=1,i=1)
        # M3
        ([ (0,0,1) ], [ (1,0,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]), # C01(k=1,i=0), C11(k=1,i=1)
        # M4
        ([ (1,1,1) ], [ (0,0,1), (1,0,-1) ], [ (0,0,1), (1,0,1) ]), # C00(k=0,i=0), C10(k=0,i=1)
        # M5
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]), # C00(k=0,i=0), C01(k=1,i=0)
        # M6
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]), # C11(k=1,i=1)
        # M7
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]), # C00(k=0,i=0)
    ]
    
    # Iterate over 5 blocks of B12
    for block in range(5):
        k_start = block * 2
        for term in strassen_terms:
            a_coeffs, b_coeffs, c_coeffs = term
            
            # Fill U
            for i, j, coeff in a_coeffs:
                U[i * m + j, r_idx] += coeff
                
            # Fill V
            for j, k_rel, coeff in b_coeffs:
                k_abs = k_start + k_rel
                V[j * p + k_abs, r_idx] += coeff
                
            # Fill W
            for k_rel, i, coeff in c_coeffs:
                k_abs = k_start + k_rel
                W[k_abs * n + i, r_idx] += coeff
                
            r_idx += 1
            
    # Part 2: A3 * B3
    # A3: i in 0..1, j=2
    # B3: j=2, k in 0..9
    # C: k in 0..9, i in 0..1
    for i in range(n):
        for k in range(p):
            U[i * m + 2, r_idx] = 1
            V[2 * p + k, r_idx] = 1
            W[k * n + i, r_idx] = 1
            r_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
