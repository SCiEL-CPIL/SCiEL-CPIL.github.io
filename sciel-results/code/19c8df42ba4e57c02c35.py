import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We use a decomposition of rank 28.
    # Split p=5 into 2, 2, 1.
    # Block 1 (k=0,1): Rank 11 (Strassen + Outer Product)
    # Block 2 (k=2,3): Rank 11 (Strassen + Outer Product)
    # Block 3 (k=4): Rank 6 (Schoolbook)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a column
    def add_col(u_indices, v_indices, w_indices, col_idx):
        for u_idx in u_indices:
            U[u_idx[0]][col_idx] += u_idx[1]
        for v_idx in v_indices:
            V[v_idx[0]][col_idx] += v_idx[1]
        for w_idx in w_indices:
            W[w_idx[0]][col_idx] += w_idx[1]
            
    # Indices for A (U): i*m + j
    # i in 0..1, j in 0..2
    A00 = 0*3+0
    A01 = 0*3+1
    A02 = 0*3+2
    A10 = 1*3+0
    A11 = 1*3+1
    A12 = 1*3+2
    
    # Indices for B (V): j*p + k
    # j in 0..2, k in 0..4
    
    # Indices for C (W): k*n + i
    # k in 0..4, i in 0..1
    
    # --- Block 1: k=0,1 ---
    # B indices for k=0,1
    B00 = 0*5+0
    B01 = 0*5+1
    B10 = 1*5+0
    B11 = 1*5+1
    B20 = 2*5+0
    B21 = 2*5+1
    
    # C indices for k=0,1
    C00 = 0*2+0
    C10 = 0*2+1
    C01 = 1*2+0
    C11 = 1*2+1
    
    col = 0
    # M1 = (A00+A11)(B00+B11)
    add_col([(A00,1),(A11,1)], [(B00,1),(B11,1)], [(C00,1),(C11,1)], col)
    col += 1
    # M2 = (A10+A11)B00
    add_col([(A10,1),(A11,1)], [(B00,1)], [(C10,1),(C11,-1)], col)
    col += 1
    # M3 = A00(B01-B11)
    add_col([(A00,1)], [(B01,1),(B11,-1)], [(C01,1),(C11,1)], col)
    col += 1
    # M4 = A11(B10-B00)
    add_col([(A11,1)], [(B10,1),(B00,-1)], [(C00,1),(C10,1)], col)
    col += 1
    # M5 = (A00+A01)B11
    add_col([(A00,1),(A01,1)], [(B11,1)], [(C00,-1),(C01,1)], col)
    col += 1
    # M6 = (A10-A11)(B01+B11)
    add_col([(A10,1),(A11,-1)], [(B01,1),(B11,1)], [(C11,1)], col)
    col += 1
    # M7 = (A01-A10)(B00+B10)
    add_col([(A01,1),(A10,-1)], [(B00,1),(B10,1)], [(C00,1)], col)
    col += 1
    # P1 = A02 B20
    add_col([(A02,1)], [(B20,1)], [(C00,1)], col)
    col += 1
    # P2 = A02 B21
    add_col([(A02,1)], [(B21,1)], [(C01,1)], col)
    col += 1
    # P3 = A12 B20
    add_col([(A12,1)], [(B20,1)], [(C10,1)], col)
    col += 1
    # P4 = A12 B21
    add_col([(A12,1)], [(B21,1)], [(C11,1)], col)
    col += 1
    
    # --- Block 2: k=2,3 ---
    # B indices for k=2,3
    B02 = 0*5+2
    B03 = 0*5+3
    B12 = 1*5+2
    B13 = 1*5+3
    B22 = 2*5+2
    B23 = 2*5+3
    
    # C indices for k=2,3
    C02 = 2*2+0
    C12 = 2*2+1
    C03 = 3*2+0
    C13 = 3*2+1
    
    # M1'
    add_col([(A00,1),(A11,1)], [(B02,1),(B13,1)], [(C02,1),(C13,1)], col)
    col += 1
    # M2'
    add_col([(A10,1),(A11,1)], [(B02,1)], [(C12,1),(C13,-1)], col)
    col += 1
    # M3'
    add_col([(A00,1)], [(B03,1),(B13,-1)], [(C03,1),(C13,1)], col)
    col += 1
    # M4'
    add_col([(A11,1)], [(B12,1),(B02,-1)], [(C02,1),(C12,1)], col)
    col += 1
    # M5'
    add_col([(A00,1),(A01,1)], [(B13,1)], [(C02,-1),(C03,1)], col)
    col += 1
    # M6'
    add_col([(A10,1),(A11,-1)], [(B03,1),(B13,1)], [(C13,1)], col)
    col += 1
    # M7'
    add_col([(A01,1),(A10,-1)], [(B02,1),(B12,1)], [(C02,1)], col)
    col += 1
    # P1'
    add_col([(A02,1)], [(B22,1)], [(C02,1)], col)
    col += 1
    # P2'
    add_col([(A02,1)], [(B23,1)], [(C03,1)], col)
    col += 1
    # P3'
    add_col([(A12,1)], [(B22,1)], [(C12,1)], col)
    col += 1
    # P4'
    add_col([(A12,1)], [(B23,1)], [(C13,1)], col)
    col += 1
    
    # --- Block 3: k=4 ---
    # B indices for k=4
    B04 = 0*5+4
    B14 = 1*5+4
    B24 = 2*5+4
    
    # C indices for k=4
    C04 = 4*2+0
    C14 = 4*2+1
    
    # Q1 = A00 B04
    add_col([(A00,1)], [(B04,1)], [(C04,1)], col)
    col += 1
    # Q2 = A01 B14
    add_col([(A01,1)], [(B14,1)], [(C04,1)], col)
    col += 1
    # Q3 = A02 B24
    add_col([(A02,1)], [(B24,1)], [(C04,1)], col)
    col += 1
    # Q4 = A10 B04
    add_col([(A10,1)], [(B04,1)], [(C14,1)], col)
    col += 1
    # Q5 = A11 B14
    add_col([(A11,1)], [(B14,1)], [(C14,1)], col)
    col += 1
    # Q6 = A12 B24
    add_col([(A12,1)], [(B24,1)], [(C14,1)], col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
