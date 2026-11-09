import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # We use the Strassen decomposition for 2x2 blocks.
    # A is 2x4, split into 2x2 blocks of 1x2.
    # B is 4x8, split into 2x2 blocks of 2x4.
    # C is 2x8, split into 2x2 blocks of 1x4.
    
    # The 7 Strassen products for the 2x2 block structure:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each product is a (1x2) * (2x4) multiplication.
    # The rank of (1x2) * (2x4) is 8.
    # Total rank = 7 * 8 = 56.
    
    products = []
    
    # Helper to add a product
    def add_product(left_coeffs, right_coeffs, c_coeffs):
        products.append({
            'left': left_coeffs,
            'right': right_coeffs,
            'c_contrib': c_coeffs
        })
        
    # Indices:
    # A is 2x4. A[i,j] is at index i*4 + j.
    # Block A00: A[0,0], A[0,1]
    # Block A01: A[0,2], A[0,3]
    # Block A10: A[1,0], A[1,1]
    # Block A11: A[1,2], A[1,3]
    
    # B is 4x8. B[i,j] is at index i*8 + j.
    # Block B00: B[0,0..3], B[1,0..3] (rows 0,1)
    # Block B01: B[0,4..7], B[1,4..7] (rows 0,1)
    # Block B10: B[2,0..3], B[3,0..3] (rows 2,3)
    # Block B11: B[2,4..7], B[3,4..7] (rows 2,3)
    
    # C is 2x8. C[i,j] is at index i*8 + j.
    # Block C00: C[0,0..3]
    # Block C01: C[0,4..7]
    # Block C10: C[1,0..3]
    # Block C11: C[1,4..7]
    
    # The (1x2) * (2x4) product generates a 1x4 vector.
    # Let L = [l0, l1] and R be 2x4.
    # Result R_res = l0*R_row0 + l1*R_row1.
    # R_res is 1x4.
    
    # We need to map the internal indices of the (1x2) * (2x4) tensor to the global indices.
    # For a specific Strassen term, we define:
    # left_coeffs: list of 2 elements. Each is a list of (global_A_idx, coeff).
    # right_coeffs: list of 8 elements. Each is a list of (global_B_idx, coeff).
    # c_coeffs: list of 4 elements. Each is a list of (global_C_idx, coeff).
    
    # M1: (A00+A11)*(B00+B11)
    # Left: A00 + A11
    # L0: A[0,0] + A[1,2]
    # L1: A[0,1] + A[1,3]
    left_M1 = [
        [(0*4+0, 1), (1*4+2, 1)],
        [(0*4+1, 1), (1*4+3, 1)]
    ]
    # Right: B00 + B11
    # R00: B[0,0] + B[2,4]
    # R01: B[0,1] + B[2,5]
    # R02: B[0,2] + B[2,6]
    # R03: B[0,3] + B[2,7]
    # R10: B[1,0] + B[3,4]
    # R11: B[1,1] + B[3,5]
    # R12: B[1,2] + B[3,6]
    # R13: B[1,3] + B[3,7]
    right_M1 = [
        [(0*8+0, 1), (2*8+4, 1)],
        [(0*8+1, 1), (2*8+5, 1)],
        [(0*8+2, 1), (2*8+6, 1)],
        [(0*8+3, 1), (2*8+7, 1)],
        [(1*8+0, 1), (3*8+4, 1)],
        [(1*8+1, 1), (3*8+5, 1)],
        [(1*8+2, 1), (3*8+6, 1)],
        [(1*8+3, 1), (3*8+7, 1)]
    ]
    # C contrib: M1 contributes to C00 and C11 (with signs)
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 adds to C00 (coeff 1) and C11 (coeff 1).
    # C00 is C[0,0..3], C11 is C[1,4..7].
    c_M1 = [
        [(0*8+0, 1), (1*8+4, 1)],
        [(0*8+1, 1), (1*8+5, 1)],
        [(0*8+2, 1), (1*8+6, 1)],
        [(0*8+3, 1), (1*8+7, 1)]
    ]
    add_product(left_M1, right_M1, c_M1)
    
    # M2: (A10+A11)*B00
    left_M2 = [
        [(1*4+0, 1), (1*4+2, 1)],
        [(1*4+1, 1), (1*4+3, 1)]
    ]
    right_M2 = [
        [(0*8+0, 1)],
        [(0*8+1, 1)],
        [(0*8+2, 1)],
        [(0*8+3, 1)],
        [(1*8+0, 1)],
        [(1*8+1, 1)],
        [(1*8+2, 1)],
        [(1*8+3, 1)]
    ]
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M2 adds to C10 (coeff 1) and C11 (coeff -1).
    # C10 is C[1,0..3], C11 is C[1,4..7].
    c_M2 = [
        [(1*8+0, 1), (1*8+4, -1)],
        [(1*8+1, 1), (1*8+5, -1)],
        [(1*8+2, 1), (1*8+6, -1)],
        [(1*8+3, 1), (1*8+7, -1)]
    ]
    add_product(left_M2, right_M2, c_M2)
    
    # M3: A00*(B01-B11)
    left_M3 = [
        [(0*4+0, 1)],
        [(0*4+1, 1)]
    ]
    right_M3 = [
        [(0*8+4, 1), (2*8+4, -1)],
        [(0*8+5, 1), (2*8+5, -1)],
        [(0*8+6, 1), (2*8+6, -1)],
        [(0*8+7, 1), (2*8+7, -1)],
        [(1*8+4, 1), (3*8+4, -1)],
        [(1*8+5, 1), (3*8+5, -1)],
        [(1*8+6, 1), (3*8+6, -1)],
        [(1*8+7, 1), (3*8+7, -1)]
    ]
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    # So M3 adds to C01 (coeff 1) and C11 (coeff 1).
    # C01 is C[0,4..7], C11 is C[1,4..7].
    c_M3 = [
        [(0*8+4, 1), (1*8+4, 1)],
        [(0*8+5, 1), (1*8+5, 1)],
        [(0*8+6, 1), (1*8+6, 1)],
        [(0*8+7, 1), (1*8+7, 1)]
    ]
    add_product(left_M3, right_M3, c_M3)
    
    # M4: A11*(B10-B00)
    left_M4 = [
        [(1*4+2, 1)],
        [(1*4+3, 1)]
    ]
    right_M4 = [
        [(2*8+0, 1), (0*8+0, -1)],
        [(2*8+1, 1), (0*8+1, -1)],
        [(2*8+2, 1), (0*8+2, -1)],
        [(2*8+3, 1), (0*8+3, -1)],
        [(3*8+0, 1), (1*8+0, -1)],
        [(3*8+1, 1), (1*8+1, -1)],
        [(3*8+2, 1), (1*8+2, -1)],
        [(3*8+3, 1), (1*8+3, -1)]
    ]
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # So M4 adds to C00 (coeff 1) and C10 (coeff 1).
    # C00 is C[0,0..3], C10 is C[1,0..3].
    c_M4 = [
        [(0*8+0, 1), (1*8+0, 1)],
        [(0*8+1, 1), (1*8+1, 1)],
        [(0*8+2, 1), (1*8+2, 1)],
        [(0*8+3, 1), (1*8+3, 1)]
    ]
    add_product(left_M4, right_M4, c_M4)
    
    # M5: (A00+A01)*B11
    left_M5 = [
        [(0*4+0, 1), (0*4+2, 1)],
        [(0*4+1, 1), (0*4+3, 1)]
    ]
    right_M5 = [
        [(2*8+4, 1)],
        [(2*8+5, 1)],
        [(2*8+6, 1)],
        [(2*8+7, 1)],
        [(3*8+4, 1)],
        [(3*8+5, 1)],
        [(3*8+6, 1)],
        [(3*8+7, 1)]
    ]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # So M5 adds to C00 (coeff -1) and C01 (coeff 1).
    # C00 is C[0,0..3], C01 is C[0,4..7].
    c_M5 = [
        [(0*8+0, -1), (0*8+4, 1)],
        [(0*8+1, -1), (0*8+5, 1)],
        [(0*8+2, -1), (0*8+6, 1)],
        [(0*8+3, -1), (0*8+7, 1)]
    ]
    add_product(left_M5, right_M5, c_M5)
    
    # M6: (A10-A00)*(B00+B01)
    left_M6 = [
        [(1*4+0, 1), (0*4+0, -1)],
        [(1*4+1, 1), (0*4+1, -1)]
    ]
    right_M6 = [
        [(0*8+0, 1), (0*8+4, 1)],
        [(0*8+1, 1), (0*8+5, 1)],
        [(0*8+2, 1), (0*8+6, 1)],
        [(0*8+3, 1), (0*8+7, 1)],
        [(1*8+0, 1), (1*8+4, 1)],
        [(1*8+1, 1), (1*8+5, 1)],
        [(1*8+2, 1), (1*8+6, 1)],
        [(1*8+3, 1), (1*8+7, 1)]
    ]
    # C11 = M1 - M2 + M3 + M6
    # So M6 adds to C11 (coeff 1).
    # C11 is C[1,4..7].
    c_M6 = [
        [(1*8+4, 1)],
        [(1*8+5, 1)],
        [(1*8+6, 1)],
        [(1*8+7, 1)]
    ]
    add_product(left_M6, right_M6, c_M6)
    
    # M7: (A01-A11)*(B10+B11)
    left_M7 = [
        [(0*4+2, 1), (1*4+2, -1)],
        [(0*4+3, 1), (1*4+3, -1)]
    ]
    right_M7 = [
        [(2*8+0, 1), (2*8+4, 1)],
        [(2*8+1, 1), (2*8+5, 1)],
        [(2*8+2, 1), (2*8+6, 1)],
        [(2*8+3, 1), (2*8+7, 1)],
        [(3*8+0, 1), (3*8+4, 1)],
        [(3*8+1, 1), (3*8+5, 1)],
        [(3*8+2, 1), (3*8+6, 1)],
        [(3*8+3, 1), (3*8+7, 1)]
    ]
    # C00 = M1 + M4 - M5 + M7
    # So M7 adds to C00 (coeff 1).
    # C00 is C[0,0..3].
    c_M7 = [
        [(0*8+0, 1)],
        [(0*8+1, 1)],
        [(0*8+2, 1)],
        [(0*8+3, 1)]
    ]
    add_product(left_M7, right_M7, c_M7)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for prod in products:
        for j in range(2):
            for k in range(4):
                # U
                for (a_idx, coeff) in prod['left'][j]:
                    U[a_idx][col] += coeff
                # V
                for (b_idx, coeff) in prod['right'][j * 4 + k]:
                    V[b_idx][col] += coeff
                # W
                for (c_idx, coeff) in prod['c_contrib'][k]:
                    W[c_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
