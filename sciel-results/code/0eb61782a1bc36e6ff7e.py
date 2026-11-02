import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val

    # Indices for A (2x3): i in 0..1, j in 0..2 -> i*3 + j
    # Indices for B (3x8): j in 0..2, k in 0..7 -> j*8 + k
    # Indices for C (8x2): k in 0..7, i in 0..1 -> k*2 + i
    
    # 4 blocks of B (3x2 each)
    for b in range(4):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # B indices for this block
        # j=0, k=k1 -> 0*8 + k1
        # j=0, k=k2 -> 0*8 + k2
        # j=1, k=k1 -> 1*8 + k1
        # j=1, k=k2 -> 1*8 + k2
        # j=2, k=k1 -> 2*8 + k1
        # j=2, k=k2 -> 2*8 + k2
        
        # A indices
        # a11 = 0*3+0=0, a12=1, a13=2
        # a21=1*3+0=3, a22=4, a23=5
        
        # C indices for this block (cols k1, k2)
        # c11 = k1*2+0, c12 = k2*2+0
        # c21 = k1*2+1, c22 = k2*2+1
        
        c11_idx = k1 * 2 + 0
        c12_idx = k2 * 2 + 0
        c21_idx = k1 * 2 + 1
        c22_idx = k2 * 2 + 1
        
        # Strassen terms for A[0:2, 0:2] * B[0:2, k1:k2]
        # Note: B rows 0,1 used for 2x2 part. Row 2 used for outer product.
        
        # M1 = (a11+a22)(b11+b22)
        # b11 = B[0, k1], b22 = B[1, k2]
        # Wait, Strassen uses B[0,0], B[0,1], B[1,0], B[1,1]
        # Here B[0,0] is B[0, k1], B[0,1] is B[0, k2]
        # B[1,0] is B[1, k1], B[1,1] is B[1, k2]
        
        b11_idx = 0 * 8 + k1
        b12_idx = 0 * 8 + k2
        b21_idx = 1 * 8 + k1
        b22_idx = 1 * 8 + k2
        
        # M1
        col = b * 11 + 0
        add_term([(0, 1), (4, 1)], [(b11_idx, 1), (b22_idx, 1)], [(c11_idx, 1), (c22_idx, 1)], col)
        
        # M2 = (a21+a22)b11
        col = b * 11 + 1
        add_term([(3, 1), (4, 1)], [(b11_idx, 1)], [(c21_idx, 1)], col)
        
        # M3 = a11(b21-b22)
        col = b * 11 + 2
        add_term([(0, 1)], [(b21_idx, 1), (b22_idx, -1)], [(c12_idx, 1)], col)
        
        # M4 = a22(b11-b12)
        col = b * 11 + 3
        add_term([(4, 1)], [(b11_idx, 1), (b12_idx, -1)], [(c11_idx, -1), (c12_idx, 1)], col)
        
        # M5 = (a11+a12)b22
        col = b * 11 + 4
        add_term([(0, 1), (1, 1)], [(b22_idx, 1)], [(c11_idx, 1), (c21_idx, 1)], col)
        
        # M6 = (a12-a11)(b11+b12)
        col = b * 11 + 5
        add_term([(1, 1), (0, -1)], [(b11_idx, 1), (b12_idx, 1)], [(c11_idx, -1), (c12_idx, 1)], col)
        
        # M7 = (a21-a22)(b21+b22)
        col = b * 11 + 6
        add_term([(3, 1), (4, -1)], [(b21_idx, 1), (b22_idx, 1)], [(c21_idx, 1), (c22_idx, -1)], col)
        
        # Outer product terms: a13*b31, a13*b32, a23*b31, a23*b32
        # b31 = B[2, k1], b32 = B[2, k2]
        b31_idx = 2 * 8 + k1
        b32_idx = 2 * 8 + k2
        
        # M8 = a13*b31
        col = b * 11 + 7
        add_term([(2, 1)], [(b31_idx, 1)], [(c11_idx, 1)], col)
        
        # M9 = a13*b32
        col = b * 11 + 8
        add_term([(2, 1)], [(b32_idx, 1)], [(c12_idx, 1)], col)
        
        # M10 = a23*b31
        col = b * 11 + 9
        add_term([(5, 1)], [(b31_idx, 1)], [(c21_idx, 1)], col)
        
        # M11 = a23*b32
        col = b * 11 + 10
        add_term([(5, 1)], [(b32_idx, 1)], [(c22_idx, 1)], col)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
