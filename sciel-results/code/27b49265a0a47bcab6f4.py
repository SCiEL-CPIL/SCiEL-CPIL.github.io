import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] = val
        for idx, val in v_indices:
            V[idx][col] = val
        for idx, val in w_indices:
            W[idx][col] = val
        col += 1

    # 1. A_2x2 * B_2x8
    for k in range(4):
        b11_idx = 16 * k
        b12_idx = 16 * k + 1
        b21_idx = 16 * k + 8
        b22_idx = 16 * k + 9
        
        c11_idx = 6 * k
        c12_idx = 6 * k + 3
        c21_idx = 6 * k + 1
        c22_idx = 6 * k + 4
        
        # M1 = (A00+A11)(B11+B22)
        add_term(
            [(0, 1), (4, 1)],
            [(b11_idx, 1), (b22_idx, 1)],
            [(c11_idx, 1), (c12_idx, 1), (c21_idx, 1), (c22_idx, -1)]
        )
        
        # M2 = (A10+A11)B11
        add_term(
            [(3, 1), (4, 1)],
            [(b11_idx, 1)],
            [(c21_idx, 1), (c22_idx, -1)]
        )
        
        # M3 = A00(B12-B22)
        add_term(
            [(0, 1)],
            [(b12_idx, 1), (b22_idx, -1)],
            [(c12_idx, 1), (c22_idx, 1)]
        )
        
        # M4 = A11(B21-B11)
        add_term(
            [(4, 1)],
            [(b21_idx, 1), (b11_idx, -1)],
            [(c11_idx, 1), (c21_idx, 1)]
        )
        
        # M5 = (A00+A01)B22
        add_term(
            [(0, 1), (1, 1)],
            [(b22_idx, 1)],
            [(c11_idx, -1), (c12_idx, 1)]
        )
        
        # M6 = (A10-A00)(B11+B12)
        add_term(
            [(3, 1), (0, -1)],
            [(b11_idx, 1), (b12_idx, 1)],
            [(c22_idx, 1)]
        )
        
        # M7 = (A01-A11)(B21+B22)
        add_term(
            [(1, 1), (4, -1)],
            [(b21_idx, 1), (b22_idx, 1)],
            [(c11_idx, 1), (c22_idx, 1)]
        )

    # 2. A_2x1 * B_1x8
    for k in range(8):
        add_term(
            [(2, 1)],
            [(16 + k, 1)],
            [(3 * k, 1)]
        )
        add_term(
            [(5, 1)],
            [(16 + k, 1)],
            [(3 * k + 1, 1)]
        )

    # 3. A_1x2 * B_2x8
    for k in range(8):
        add_term(
            [(6, 1)],
            [(8 * k, 1)],
            [(3 * k + 2, 1)]
        )
        add_term(
            [(7, 1)],
            [(8 * k + 1, 1)],
            [(3 * k + 2, 1)]
        )

    # 4. A_1x1 * B_1x8
    for k in range(8):
        add_term(
            [(8, 1)],
            [(16 + k, 1)],
            [(3 * k + 2, 1)]
        )

    return {"U": U, "V": V, "W": W}
