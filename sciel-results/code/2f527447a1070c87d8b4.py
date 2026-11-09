import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_mult(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # u_idx, v_idx are indices in flattened A and B
        # w_idx is index in flattened C
        # We add a new column to U, V, W
        pass # Placeholder

    # We will build columns one by one.
    # Rank is 36.
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1
        
    # Indices:
    # A: i * m + j. i in 0..1, j in 0..3.
    # B: j * p + k. j in 0..3, k in 0..4.
    # C: k * n + i. k in 0..4, i in 0..1.
    
    # Strassen for A_block (2x2) * B_block (2x2)
    # A_block indices: a00, a01, a10, a11
    # B_block indices: b00, b01, b10, b11
    # C_block indices: c00, c01, c10, c11
    
    def strassen_2x2(a_base, b_base, c_base):
        # a_base: (i, j_base) -> index in A
        # b_base: (j_base, k) -> index in B
        # c_base: (k, i) -> index in C
        # We assume a_base(i, j) returns the flat index for A[i, j]
        # b_base(j, k) returns the flat index for B[j, k]
        # c_base(k, i) returns the flat index for C[k, i]
        
        # M1 = A00 * (B01 - B11)
        add_term([a_base(0, 0)], [1], [b_base(0, 1), b_base(1, 1)], [1, -1], [c_base(1, 0), c_base(1, 1)], [1, 1])
        # M2 = (A00 + A10) * B11
        add_term([a_base(0, 0), a_base(1, 0)], [1, 1], [b_base(1, 1)], [1], [c_base(0, 0), c_base(0, 1)], [-1, 1])
        # M3 = (A01 + A11) * B00
        add_term([a_base(0, 1), a_base(1, 1)], [1, 1], [b_base(0, 0)], [1], [c_base(0, 1), c_base(1, 0)], [1, 1])
        # M4 = A11 * (B10 - B00)
        add_term([a_base(1, 1)], [1], [b_base(1, 0), b_base(0, 0)], [1, -1], [c_base(0, 0), c_base(1, 1)], [1, 1])
        # M5 = (A00 + A11) * (B00 + B11)
        add_term([a_base(0, 0), a_base(1, 1)], [1, 1], [b_base(0, 0), b_base(1, 1)], [1, 1], [c_base(0, 0), c_base(0, 1), c_base(1, 0), c_base(1, 1)], [1, 1, -1, 1])
        # M6 = (A01 - A11) * (B10 + B11)
        add_term([a_base(0, 1), a_base(1, 1)], [1, -1], [b_base(1, 0), b_base(1, 1)], [1, 1], [c_base(0, 0), c_base(0, 1)], [1, 1])
        # M7 = (A00 - A10) * (B00 + B01)
        add_term([a_base(0, 0), a_base(1, 0)], [1, -1], [b_base(0, 0), b_base(0, 1)], [1, 1], [c_base(1, 0), c_base(1, 1)], [-1, 1])

    def schoolbook_2x2_1(a_base, b_base, c_base):
        # A 2x2, B 2x1, C 2x1
        # K1 = A00 * B00
        add_term([a_base(0, 0)], [1], [b_base(0, 0)], [1], [c_base(0, 0)], [1])
        # K2 = A01 * B10
        add_term([a_base(0, 1)], [1], [b_base(1, 0)], [1], [c_base(0, 0)], [1])
        # K3 = A10 * B00
        add_term([a_base(1, 0)], [1], [b_base(0, 0)], [1], [c_base(1, 0)], [1])
        # K4 = A11 * B10
        add_term([a_base(1, 1)], [1], [b_base(1, 0)], [1], [c_base(1, 0)], [1])

    # Block 1: A cols 0,1. B rows 0,1.
    # A indices: i*4 + j. j in 0,1.
    # B indices: j*5 + k. j in 0,1.
    
    # Sub-block 1.1: C cols 0,1. B cols 0,1.
    def a1_base(i, j): return i * 4 + j
    def b11_base(j, k): return j * 5 + k
    def c11_base(k, i): return k * 2 + i
    strassen_2x2(a1_base, b11_base, c11_base)
    
    # Sub-block 1.2: C cols 2,3. B cols 2,3.
    def b12_base(j, k): return j * 5 + (k + 2)
    def c12_base(k, i): return (k + 2) * 2 + i
    strassen_2x2(a1_base, b12_base, c12_base)
    
    # Sub-block 1.3: C col 4. B col 4.
    def b13_base(j, k): return j * 5 + 4
    def c13_base(k, i): return 4 * 2 + i
    schoolbook_2x2_1(a1_base, b13_base, c13_base)
    
    # Block 2: A cols 2,3. B rows 2,3.
    def a2_base(i, j): return i * 4 + (j + 2)
    def b21_base(j, k): return (j + 2) * 5 + k
    def c21_base(k, i): return k * 2 + i
    strassen_2x2(a2_base, b21_base, c21_base)
    
    # Sub-block 2.2: C cols 2,3. B cols 2,3.
    def b22_base(j, k): return (j + 2) * 5 + (k + 2)
    def c22_base(k, i): return (k + 2) * 2 + i
    strassen_2x2(a2_base, b22_base, c22_base)
    
    # Sub-block 2.3: C col 4. B col 4.
    def b23_base(j, k): return (j + 2) * 5 + 4
    def c23_base(k, i): return 4 * 2 + i
    schoolbook_2x2_1(a2_base, b23_base, c23_base)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
