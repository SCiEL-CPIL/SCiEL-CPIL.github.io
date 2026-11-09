import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 blocks of size 3x3
    # Blocks are indexed (br, bc) in {0,1}x{0,1}
    # A[br, bc], B[br, bc], C[br, bc]
    
    # Define the 7 products
    # Each product is defined by:
    # A_coeffs: dict {(br, bc): coeff}
    # B_coeffs: dict {(br, bc): coeff}
    # C_coeffs: dict {(br, bc): coeff}
    
    products = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += M1, C11 += M1
    products.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A10 + A11)B00
    # C10 += M2, C11 -= M2
    products.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = A00(B01 - B11)
    # C01 += M3, C11 += M3
    products.append({
        'A': {(0,0): 1},
        'B': {(0,1): 1, (1,1): -1},
        'C': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = A11(B10 - B00)
    # C00 += M4, C10 += M4
    products.append({
        'A': {(1,1): 1},
        'B': {(1,0): 1, (0,0): -1},
        'C': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (A00 + A01)B11
    # C00 -= M5, C01 += M5
    products.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (A10 - A00)(B00 + B01)
    # C11 += M6
    products.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,1): 1}
    })
    
    # M7 = (A01 - A11)(B10 + B11)
    # C00 += M7
    products.append({
        'A': {(0,1): 1, (1,1): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(0,0): 1}
    })
    
    col = 0
    for prod in products:
        A_coeffs = prod['A']
        B_coeffs = prod['B']
        C_coeffs = prod['C']
        
        # For each local index i, j, k in 3x3 block
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Update U
                    for (br, bc), coeff in A_coeffs.items():
                        # Global row in A: 3*br + i
                        # Global col in A: 3*bc + j
                        row_u = (3 * br + i) * m + (3 * bc + j)
                        U[row_u][col] += coeff
                    
                    # Update V
                    for (br, bc), coeff in B_coeffs.items():
                        # Global row in B: 3*br + j
                        # Global col in B: 3*bc + k
                        row_v = (3 * br + j) * p + (3 * bc + k)
                        V[row_v][col] += coeff
                    
                    # Update W
                    for (br, bc), coeff in C_coeffs.items():
                        # Global row in C: 3*br + i
                        # Global col in C: 3*bc + k
                        # W index: k_global * n + i_global
                        row_w = (3 * bc + k) * n + (3 * br + i)
                        W[row_w][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
