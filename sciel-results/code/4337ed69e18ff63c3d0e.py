import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 2: A_R (2x1) * B_R (1x5) -> Rank 10
    # A_R is col 2 of A. B_R is row 2 of B.
    # C[i, k] = A[i, 2] * B[2, k]
    col = 0
    for i in range(n):
        for k in range(p):
            # U index: i * m + 2
            U[i * m + 2, col] = 1.0
            # V index: 2 * p + k
            V[2 * p + k, col] = 1.0
            # W index: k * n + i
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 1: A_L (2x2) * B_L (2x5) -> Rank 18
    # A_L is cols 0,1 of A. B_L is rows 0,1 of B.
    
    # Subpart 1a: Cols 0,1 of B_L -> Strassen (Rank 7)
    # B_L cols 0,1 correspond to B cols 0,1.
    # A_L entries: a00, a01, a10, a11.
    # B_L entries: b00, b01, b10, b11.
    
    # Strassen formulas for C = A * B
    # M1 = (a11 - a01) * (b00 + b11)
    # M2 = (a00 + a01) * b11
    # M3 = a00 * (b10 - b00)
    # M4 = a11 * (b00 + b10)
    # M5 = (a00 + a11) * (b11 - b01)
    # M6 = (a10 + a11) * b00
    # M7 = (a01 - a10) * (b01 + b10)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M5
    # C11 = M1 - M4 + M5 + M7
    
    # Map to U, V, W
    # U indices: 0->a00, 1->a01, 3->a10, 4->a11 (since m=3, i*m+j)
    # V indices: 0->b00, 1->b01, 5->b10, 6->b11 (since p=5, j*p+k)
    # W indices: 0->c00, 1->c10, 2->c01, 3->c11 (since n=2, k*n+i)
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals, c_idx):
        for idx, val in u_vals.items():
            U[idx, c_idx] += val
        for idx, val in v_vals.items():
            V[idx, c_idx] += val
        for idx, val in w_vals.items():
            W[idx, c_idx] += val
            
    # M1
    add_term({4: 1, 1: -1}, {0: 1, 6: 1}, {2: 1, 3: 1}, col)
    col += 1
    # M2
    add_term({0: 1, 1: 1}, {6: 1}, {0: -1, 2: 1}, col)
    col += 1
    # M3
    add_term({0: 1}, {5: 1, 0: -1}, {1: 1}, col)
    col += 1
    # M4
    add_term({4: 1}, {0: 1, 5: 1}, {0: 1, 3: -1}, col)
    col += 1
    # M5
    add_term({0: 1, 4: 1}, {6: 1, 1: -1}, {0: 1, 1: 1, 3: 1}, col)
    col += 1
    # M6
    add_term({3: 1, 4: 1}, {0: 1}, {0: 1}, col)
    col += 1
    # M7
    add_term({1: 1, 3: -1}, {1: 1, 5: 1}, {3: 1}, col)
    col += 1
    
    # Subpart 1b: Cols 2,3 of B_L -> Strassen (Rank 7)
    # B_L cols 2,3 correspond to B cols 2,3.
    # V indices: 2->b02, 3->b03, 7->b12, 8->b13
    # W indices: 4->c02, 5->c12, 6->c03, 7->c13
    
    # M1
    add_term({4: 1, 1: -1}, {2: 1, 8: 1}, {6: 1, 7: 1}, col)
    col += 1
    # M2
    add_term({0: 1, 1: 1}, {8: 1}, {4: -1, 6: 1}, col)
    col += 1
    # M3
    add_term({0: 1}, {7: 1, 2: -1}, {5: 1}, col)
    col += 1
    # M4
    add_term({4: 1}, {2: 1, 7: 1}, {4: 1, 7: -1}, col)
    col += 1
    # M5
    add_term({0: 1, 4: 1}, {8: 1, 3: -1}, {4: 1, 5: 1, 7: 1}, col)
    col += 1
    # M6
    add_term({3: 1, 4: 1}, {2: 1}, {4: 1}, col)
    col += 1
    # M7
    add_term({1: 1, 3: -1}, {3: 1, 7: 1}, {7: 1}, col)
    col += 1
    
    # Subpart 1c: Col 4 of B_L -> Schoolbook (Rank 4)
    # B_L col 4 corresponds to B col 4.
    # V indices: 4->b04, 9->b14
    # W indices: 8->c04, 9->c14
    
    # a00 * b04 -> c04
    add_term({0: 1}, {4: 1}, {8: 1}, col)
    col += 1
    # a01 * b14 -> c04
    add_term({1: 1}, {9: 1}, {8: 1}, col)
    col += 1
    # a10 * b04 -> c14
    add_term({3: 1}, {4: 1}, {9: 1}, col)
    col += 1
    # a11 * b14 -> c14
    add_term({4: 1}, {9: 1}, {9: 1}, col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
