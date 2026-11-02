import time
import sys

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,2,8} into 4 independent M_{2,2,2} tensors.
    # Each M_{2,2,2} is decomposed using Strassen's algorithm (rank 7).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for the 7 products
    # Each product is defined by:
    # u: list of (i, j, coeff) for U
    # v: list of (j, k_offset, coeff) for V (k_offset is 0 for k0, 1 for k1)
    # w: list of (k_offset, i, coeff) for W
    
    # Product 1: (A00 + A11)(B0k0 + B1k1)
    # U: (0,0,1), (1,1,1)
    # V: (0,0,1), (1,1,1)
    # W: (0,0,1), (1,1,1)
    p1_u = [(0, 0, 1), (1, 1, 1)]
    p1_v = [(0, 0, 1), (1, 1, 1)]
    p1_w = [(0, 0, 1), (1, 1, 1)]
    
    # Product 2: (A10 + A11) B0k0
    # U: (1,0,1), (1,1,1)
    # V: (0,0,1)
    # W: (0,1,1)
    p2_u = [(1, 0, 1), (1, 1, 1)]
    p2_v = [(0, 0, 1)]
    p2_w = [(0, 1, 1)]
    
    # Product 3: A00 (B0k1 - B1k1)
    # U: (0,0,1)
    # V: (0,1,1), (1,1,-1)
    # W: (1,0,1)
    p3_u = [(0, 0, 1)]
    p3_v = [(0, 1, 1), (1, 1, -1)]
    p3_w = [(1, 0, 1)]
    
    # Product 4: A11 (B1k0 - B0k0)
    # U: (1,1,1)
    # V: (1,0,1), (0,0,-1)
    # W: (0,1,1), (0,0,1)
    p4_u = [(1, 1, 1)]
    p4_v = [(1, 0, 1), (0, 0, -1)]
    p4_w = [(0, 1, 1), (0, 0, 1)]
    
    # Product 5: (A00 + A01) B1k1
    # U: (0,0,1), (0,1,1)
    # V: (1,1,1)
    # W: (0,0,-1), (1,0,1)
    p5_u = [(0, 0, 1), (0, 1, 1)]
    p5_v = [(1, 1, 1)]
    p5_w = [(0, 0, -1), (1, 0, 1)]
    
    # Product 6: (A10 - A00) (B0k0 + B0k1)
    # U: (1,0,1), (0,0,-1)
    # V: (0,0,1), (0,1,1)
    # W: (1,1,1)
    p6_u = [(1, 0, 1), (0, 0, -1)]
    p6_v = [(0, 0, 1), (0, 1, 1)]
    p6_w = [(1, 1, 1)]
    
    # Product 7: (A01 - A11) (B1k0 + B1k1)
    # U: (0,1,1), (1,1,-1)
    # V: (1,0,1), (1,1,1)
    # W: (0,0,1), (1,1,1)
    p7_u = [(0, 1, 1), (1, 1, -1)]
    p7_v = [(1, 0, 1), (1, 1, 1)]
    p7_w = [(0, 0, 1), (1, 1, 1)]
    
    products = [
        (p1_u, p1_v, p1_w),
        (p2_u, p2_v, p2_w),
        (p3_u, p3_v, p3_w),
        (p4_u, p4_v, p4_w),
        (p5_u, p5_v, p5_w),
        (p6_u, p6_v, p6_w),
        (p7_u, p7_v, p7_w)
    ]
    
    col = 0
    for r in range(4):
        k0 = 2 * r
        k1 = 2 * r + 1
        for u_coeff, v_coeff, w_coeff in products:
            # Fill U
            for i, j, val in u_coeff:
                U[i * m + j][col] += val
            
            # Fill V
            for j, k_off, val in v_coeff:
                k = k0 if k_off == 0 else k1
                V[j * p + k][col] += val
            
            # Fill W
            for k_off, i, val in w_coeff:
                k = k0 if k_off == 0 else k1
                W[k * n + i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
