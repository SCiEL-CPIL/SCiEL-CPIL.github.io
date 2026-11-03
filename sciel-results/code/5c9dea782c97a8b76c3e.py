import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We expect rank 42
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen products
    # Y is 2x2, Z is 2x2
    # M1: (Y00+Y11)(Z00+Z11) -> X00, X11
    # M2: (Y10+Y11)Z00 -> X10
    # M3: Y00(Z10-Z11) -> X01
    # M4: Y11(Z01-Z00) -> X00
    # M5: (Y00+Y01)Z11 -> X00, X01
    # M6: (Y10-Y00)(Z00+Z01) -> X11
    # M7: (Y01-Y11)(Z10+Z11) -> X00, X11
    
    # We need to map local (r,c) to global indices for A, B, C blocks
    # A block: rows i0..i0+1, cols j0..j0+1
    # B block: rows i0..i0+1, cols j0..j0+1
    # C block: rows i0..i0+1, cols j0..j0+1
    
    def add_strassen_term(a_i, a_j, b_i, b_j, c_i, c_j):
        nonlocal col
        # a_i, a_j: top-left row/col of A block
        # b_i, b_j: top-left row/col of B block
        # c_i, c_j: top-left row/col of C block
        
        # Helper to get global index for U (A)
        def u_idx(r, c):
            return (a_i + r) * m + (a_j + c)
            
        # Helper to get global index for V (B)
        def v_idx(r, c):
            return (b_i + r) * p + (b_j + c)
            
        # Helper to get global index for W (C)
        def w_idx(r, c):
            return (c_j + c) * n + (c_i + r)
            
        # 7 terms
        # M1
        U[u_idx(0,0)][col] = 1
        U[u_idx(1,1)][col] = 1
        V[v_idx(0,0)][col] = 1
        V[v_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = 1
        W[w_idx(1,1)][col] = 1
        col += 1
        
        # M2
        U[u_idx(1,0)][col] = 1
        U[u_idx(1,1)][col] = 1
        V[v_idx(0,0)][col] = 1
        W[w_idx(1,0)][col] = 1
        col += 1
        
        # M3
        U[u_idx(0,0)][col] = 1
        V[v_idx(1,0)][col] = 1
        V[v_idx(1,1)][col] = -1
        W[w_idx(0,1)][col] = 1
        col += 1
        
        # M4
        U[u_idx(1,1)][col] = 1
        V[v_idx(0,1)][col] = 1
        V[v_idx(0,0)][col] = -1
        W[w_idx(0,0)][col] = 1
        col += 1
        
        # M5
        U[u_idx(0,0)][col] = 1
        U[u_idx(0,1)][col] = 1
        V[v_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = -1
        W[w_idx(0,1)][col] = 1
        col += 1
        
        # M6
        U[u_idx(1,0)][col] = 1
        U[u_idx(0,0)][col] = -1
        V[v_idx(0,0)][col] = 1
        V[v_idx(0,1)][col] = 1
        W[w_idx(1,1)][col] = 1
        col += 1
        
        # M7
        U[u_idx(0,1)][col] = 1
        U[u_idx(1,1)][col] = -1
        V[v_idx(1,0)][col] = 1
        V[v_idx(1,1)][col] = 1
        W[w_idx(0,0)][col] = 1
        W[w_idx(1,1)][col] = 1
        col += 1

    # Part A: C[:, 0:4] = A B[:, 0:4]
    # A0 (cols 0-1) * B00 (rows 0-1, cols 0-1) -> C0 (cols 0-1)
    add_strassen_term(0, 0, 0, 0, 0, 0)
    # A1 (cols 2-3) * B10 (rows 2-3, cols 0-1) -> C0 (cols 0-1)
    add_strassen_term(0, 2, 2, 0, 0, 0)
    # A0 (cols 0-1) * B01 (rows 0-1, cols 2-3) -> C1 (cols 2-3)
    add_strassen_term(0, 0, 0, 2, 0, 2)
    # A1 (cols 2-3) * B11 (rows 2-3, cols 2-3) -> C1 (cols 2-3)
    add_strassen_term(0, 2, 2, 2, 0, 2)
    
    # Part B: C[:, 4:6] = A B[:, 4:6]
    # A0 (cols 0-1) * B02 (rows 0-1, cols 4-5) -> C2 (cols 4-5)
    add_strassen_term(0, 0, 0, 4, 0, 4)
    # A1 (cols 2-3) * B12 (rows 2-3, cols 4-5) -> C2 (cols 4-5)
    add_strassen_term(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
