import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    
    # U: (n*m, rank) = (9, 34)
    # V: (m*p, rank) = (12, 34)
    # W: (p*n, rank) = (12, 34)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_indices: list of (idx, coeff) for A
    # v_indices: list of (idx, coeff) for B
    # w_indices: list of (idx, coeff) for C
    def add_term(col, u_indices, v_indices, w_indices):
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff

    # Indices for A (3x3): i*3 + j
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    # A20=6, A21=7, A22=8
    
    # Indices for B (3x4): j*4 + k
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    # B20=8, B21=9, B22=10, B23=11
    
    # Indices for C (3x4): k*3 + i
    # C00=0, C10=1, C20=2 (Col 0: i=0,1,2 -> k=0)
    # C01=3, C11=4, C21=5 (Col 1: i=0,1,2 -> k=1)
    # C02=6, C12=7, C22=8 (Col 2: i=0,1,2 -> k=2)
    # C03=9, C13=10, C23=11 (Col 3: i=0,1,2 -> k=3)
    
    # Term 1: P1 = (A00 + A11)(B00 + B10)
    # Contributes to C00, C01, C10, C11
    # C00 += P1
    # C10 += P1 (from C22 of top block? No, C11 of top block is A11*B11? No)
    # Let's re-verify Strassen mapping.
    # C_top = [[C00, C01], [C10, C11]]
    # C00 = P1 - P4 - P6 + P7
    # C01 = P5 - P7
    # C10 = P2 - P4 - P7
    # C11 = P1 - P2 + P3
    
    col = 0
    # P1
    add_term(col, [(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (1, 1), (3, 1), (4, 1)])
    col += 1
    
    # P2 = (A10 + A11) B00
    # C10 += P2
    # C11 -= P2
    add_term(col, [(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    col += 1
    
    # P3 = A00 (B10 + B11)
    # C11 += P3
    add_term(col, [(0, 1)], [(4, 1), (5, 1)], [(4, 1)])
    col += 1
    
    # P4 = A11 (B00 - B10)
    # C00 -= P4
    # C10 -= P4
    # C11 -= P4
    add_term(col, [(4, 1)], [(0, 1), (4, -1)], [(0, -1), (1, -1), (4, -1)])
    col += 1
    
    # P5 = (A00 + A01) B11
    # C01 += P5
    add_term(col, [(0, 1), (1, 1)], [(5, 1)], [(3, 1)])
    col += 1
    
    # P6 = (A10 - A00) (B00 + B01)
    # C00 -= P6
    add_term(col, [(3, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1)])
    col += 1
    
    # P7 = (A01 - A11) (B10 + B11)
    # C00 += P7
    # C01 -= P7
    # C10 -= P7
    add_term(col, [(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1), (3, -1), (1, -1)])
    col += 1
    
    # Term 2: Q1-Q7 for B_top,1 (cols 2,3)
    # B02=2, B03=3, B12=6, B13=7
    # C02=6, C03=9, C12=7, C13=10
    
    # Q1 = (A00 + A11)(B02 + B12)
    add_term(col, [(0, 1), (4, 1)], [(2, 1), (6, 1)], [(6, 1), (7, 1), (9, 1), (10, 1)])
    col += 1
    
    # Q2 = (A10 + A11) B02
    add_term(col, [(3, 1), (4, 1)], [(2, 1)], [(7, 1), (10, -1)])
    col += 1
    
    # Q3 = A00 (B12 + B13)
    add_term(col, [(0, 1)], [(6, 1), (7, 1)], [(10, 1)])
    col += 1
    
    # Q4 = A11 (B02 - B12)
    add_term(col, [(4, 1)], [(2, 1), (6, -1)], [(6, -1), (7, -1), (10, -1)])
    col += 1
    
    # Q5 = (A00 + A01) B13
    add_term(col, [(0, 1), (1, 1)], [(7, 1)], [(9, 1)])
    col += 1
    
    # Q6 = (A10 - A00) (B02 + B03)
    add_term(col, [(3, 1), (0, -1)], [(2, 1), (3, 1)], [(6, -1)])
    col += 1
    
    # Q7 = (A01 - A11) (B12 + B13)
    add_term(col, [(1, 1), (4, -1)], [(6, 1), (7, 1)], [(6, 1), (9, -1), (7, -1)])
    col += 1
    
    # Term 3: R1-R8
    # R1 = A02 B20 -> C00
    add_term(col, [(2, 1)], [(8, 1)], [(0, 1)])
    col += 1
    # R2 = A02 B21 -> C01
    add_term(col, [(2, 1)], [(9, 1)], [(3, 1)])
    col += 1
    # R3 = A02 B22 -> C02
    add_term(col, [(2, 1)], [(10, 1)], [(6, 1)])
    col += 1
    # R4 = A02 B23 -> C03
    add_term(col, [(2, 1)], [(11, 1)], [(9, 1)])
    col += 1
    # R5 = A12 B20 -> C10
    add_term(col, [(5, 1)], [(8, 1)], [(1, 1)])
    col += 1
    # R6 = A12 B21 -> C11
    add_term(col, [(5, 1)], [(9, 1)], [(4, 1)])
    col += 1
    # R7 = A12 B22 -> C12
    add_term(col, [(5, 1)], [(10, 1)], [(7, 1)])
    col += 1
    # R8 = A12 B23 -> C13
    add_term(col, [(5, 1)], [(11, 1)], [(10, 1)])
    col += 1
    
    # Term 4: S1-S8
    # S1 = A20 B00 -> C20
    add_term(col, [(6, 1)], [(0, 1)], [(2, 1)])
    col += 1
    # S2 = A20 B01 -> C21
    add_term(col, [(6, 1)], [(1, 1)], [(5, 1)])
    col += 1
    # S3 = A20 B02 -> C22
    add_term(col, [(6, 1)], [(2, 1)], [(8, 1)])
    col += 1
    # S4 = A20 B03 -> C23
    add_term(col, [(6, 1)], [(3, 1)], [(11, 1)])
    col += 1
    # S5 = A21 B10 -> C20
    add_term(col, [(7, 1)], [(4, 1)], [(2, 1)])
    col += 1
    # S6 = A21 B11 -> C21
    add_term(col, [(7, 1)], [(5, 1)], [(5, 1)])
    col += 1
    # S7 = A21 B12 -> C22
    add_term(col, [(7, 1)], [(6, 1)], [(8, 1)])
    col += 1
    # S8 = A21 B13 -> C23
    add_term(col, [(7, 1)], [(7, 1)], [(11, 1)])
    col += 1
    
    # Term 5: T1-T4
    # T1 = A22 B20 -> C20
    add_term(col, [(8, 1)], [(8, 1)], [(2, 1)])
    col += 1
    # T2 = A22 B21 -> C21
    add_term(col, [(8, 1)], [(9, 1)], [(5, 1)])
    col += 1
    # T3 = A22 B22 -> C22
    add_term(col, [(8, 1)], [(10, 1)], [(8, 1)])
    col += 1
    # T4 = A22 B23 -> C23
    add_term(col, [(8, 1)], [(11, 1)], [(11, 1)])
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
