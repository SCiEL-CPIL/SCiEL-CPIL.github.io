import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)] # 6 x 23
    V = [[0.0] * rank for _ in range(m * p)] # 12 x 23
    W = [[0.0] * rank for _ in range(p * n)] # 8 x 23
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
        col += 1

    # Sub-problem 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i in {0,1}, j in {0,1}. U idx = i*3 + j
    # B indices: j in {0,1}, k in {0,1}. V idx = j*4 + k
    # C indices: i in {0,1}, k in {0,1}. W idx = k*2 + i
    
    # M1 = (A00 + A11) * (B00 + B11)
    add_term(
        [(0*3+0, 1), (1*3+1, 1)],
        [(0*4+0, 1), (1*4+1, 1)],
        [(0*2+0, 1), (1*2+1, 1)] # C00, C11
    )
    # Wait, Strassen sums into C.
    # C00 += M1, C11 += M1
    # My add_term just adds to the specific indices.
    # I need to add M1 to C00 and C11.
    # So w_indices should be [(0, 1), (3, 1)]?
    # W index for C00 (i=0, k=0) is 0*2+0 = 0.
    # W index for C11 (i=1, k=1) is 1*2+1 = 3.
    
    # Let's restart the term additions properly.
    
    # Reset col
    col = 0
    
    # M1
    add_term(
        [(0, 1), (4, 1)], # A00, A11
        [(0, 1), (5, 1)], # B00, B11
        [(0, 1), (3, 1)]  # C00, C11
    )
    
    # M2 = (A10 + A11) * B00
    add_term(
        [(3, 1), (4, 1)], # A10, A11
        [(0, 1)],         # B00
        [(2, 1), (3, -1)] # C10 += M2, C11 -= M2
    )
    # C10 is i=1, k=0 -> idx 2. C11 is i=1, k=1 -> idx 3.
    # Formula: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # So M2 adds to C10, subtracts from C11.
    
    # M3 = A00 * (B01 - B11)
    add_term(
        [(0, 1)],         # A00
        [(1, 1), (5, -1)],# B01, B11
        [(1, 1), (3, 1)]  # C01 += M3, C11 += M3
    )
    # C01 is i=0, k=1 -> idx 1. C11 is idx 3.
    
    # M4 = A11 * (B10 - B00)
    add_term(
        [(4, 1)],         # A11
        [(4, 1), (0, -1)],# B10, B00
        [(0, 1), (2, 1)]  # C00 += M4, C10 += M4
    )
    # C00 idx 0. C10 idx 2.
    
    # M5 = (A00 + A01) * B11
    add_term(
        [(0, 1), (1, 1)], # A00, A01
        [(5, 1)],         # B11
        [(0, -1), (1, 1)] # C00 -= M5, C01 += M5
    )
    # C00 idx 0. C01 idx 1.
    
    # M6 = (A10 - A00) * (B00 + B01)
    add_term(
        [(3, 1), (0, -1)], # A10, A00
        [(0, 1), (1, 1)],  # B00, B01
        [(3, 1)]           # C11 += M6
    )
    # C11 idx 3.
    
    # M7 = (A01 - A11) * (B10 + B11)
    add_term(
        [(1, 1), (4, -1)], # A01, A11
        [(4, 1), (5, 1)],  # B10, B11
        [(0, 1), (1, -1)]  # C00 += M7, C01 -= M7
    )
    # C00 idx 0. C01 idx 1.
    
    # Sub-problem 2: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # B col 2. V indices: j*4 + 2. j in {0,1} -> 2, 6.
    # C col 2. W indices: k*2 + i. k=2 -> 4+i. i in {0,1} -> 4, 5.
    
    # T1 = A00 * B02
    add_term([(0, 1)], [(2, 1)], [(4, 1)])
    # T2 = A01 * B12
    add_term([(1, 1)], [(6, 1)], [(4, 1)])
    # T3 = A10 * B02
    add_term([(3, 1)], [(2, 1)], [(5, 1)])
    # T4 = A11 * B12
    add_term([(4, 1)], [(6, 1)], [(5, 1)])
    
    # Sub-problem 3: A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # A col 2. U indices: i*3 + 2. i in {0,1} -> 2, 5.
    # B row 2. V indices: 2*4 + k. k in {0,1,2} -> 8, 9, 10.
    # C all k. W indices: k*2 + i.
    
    # T5 = A02 * B20 -> C00
    add_term([(2, 1)], [(8, 1)], [(0, 1)])
    # T6 = A02 * B21 -> C01
    add_term([(2, 1)], [(9, 1)], [(1, 1)])
    # T7 = A02 * B22 -> C02
    add_term([(2, 1)], [(10, 1)], [(4, 1)])
    # T8 = A12 * B20 -> C10
    add_term([(5, 1)], [(8, 1)], [(2, 1)])
    # T9 = A12 * B21 -> C11
    add_term([(5, 1)], [(9, 1)], [(3, 1)])
    # T10 = A12 * B22 -> C12
    add_term([(5, 1)], [(10, 1)], [(5, 1)])
    
    # Sub-problem 4: A[0:2, 0:3] * B[0:3, 3] -> C[0:2, 3]
    # B col 3. V indices: j*4 + 3. j in {0,1,2} -> 3, 7, 11.
    # C col 3. W indices: k*2 + i. k=3 -> 6+i. i in {0,1} -> 6, 7.
    
    # T11 = A00 * B03
    add_term([(0, 1)], [(3, 1)], [(6, 1)])
    # T12 = A01 * B13
    add_term([(1, 1)], [(7, 1)], [(6, 1)])
    # T13 = A02 * B23
    add_term([(2, 1)], [(11, 1)], [(6, 1)])
    # T14 = A10 * B03
    add_term([(3, 1)], [(3, 1)], [(7, 1)])
    # T15 = A11 * B13
    add_term([(4, 1)], [(7, 1)], [(7, 1)])
    # T16 = A12 * B23
    add_term([(5, 1)], [(11, 1)], [(7, 1)])
    
    return {"U": U, "V": V, "W": W}
