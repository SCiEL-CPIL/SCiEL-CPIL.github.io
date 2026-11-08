import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Rank 29 construction:
    # Split m=4 into m1=2, m2=1.
    # Split p=7 into p1=6, p2=1.
    # 
    # Block 1: m1=2, p1=6. Split p1 into three 2s.
    #   3 * Strassen(2,2,2) = 3 * 7 = 21.
    # Block 2: m2=1, p1=6. Split p1 into three 2s.
    #   3 * Rank(2,1,2) = 3 * 1 = 3.
    # Block 3: m1=2, p2=1.
    #   Rank(2,2,1) = 4.
    # Block 4: m2=1, p2=1.
    #   Rank(2,1,1) = 1.
    # Total = 21 + 3 + 4 + 1 = 29.
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # A (2x2 block at row i0, col j0)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B (2x2 block at row j0, col k0)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # W (2x2 block at row k0, col i0)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00, C11
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10, -C11
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01, C11
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00, C10
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h -> -C00, C01
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_rank1_212_block(i0, j0, k0):
        nonlocal col
        # A (2x1 block)
        a_idx = i0 * m + j0
        c_idx = (i0 + 1) * m + j0
        
        # B (1x2 block)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        
        # W (2x2 block)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # C = A * B (outer product)
        U[a_idx][col] += 1
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c00_idx][col] += 1
        W[c01_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

    def add_rank1_211_block(i0, j0, k0):
        nonlocal col
        # A (2x1 block)
        a_idx = i0 * m + j0
        c_idx = (i0 + 1) * m + j0
        
        # B (1x1 block)
        e_idx = j0 * p + k0
        
        # W (2x1 block)
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # C = A * B
        U[a_idx][col] += 1
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        # A (2x2 block)
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        # B (2x1 block)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        # W (2x1 block)
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C00
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P2 = b*g -> C00
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        # P3 = c*e -> C10
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P4 = d*g -> C10
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Apply blocks
    # m1 = 2 (j=0,1), m2 = 1 (j=2)
    # p1 = 6 (k=0..5), p2 = 1 (k=6)
    
    # Block 1: m1, p1 (3x Strassen)
    for k_start in [0, 2, 4]:
        add_strassen_block(0, 0, k_start)
        
    # Block 2: m2, p1 (3x Rank1)
    for k_start in [0, 2, 4]:
        add_rank1_212_block(0, 2, k_start)
        
    # Block 3: m1, p2 (1x Schoolbook 221)
    add_schoolbook_221_block(0, 0, 6)
    
    # Block 4: m2, p2 (1x Rank1)
    add_rank1_211_block(0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
