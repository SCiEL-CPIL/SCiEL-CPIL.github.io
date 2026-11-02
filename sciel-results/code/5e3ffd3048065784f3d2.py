import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    def add_terms(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for u_idx, val in u_dict.items():
            U[u_idx][col] = val
        for v_idx, val in v_dict.items():
            V[v_idx][col] = val
        for w_idx, val in w_dict.items():
            W[w_idx][col] = val
        col += 1

    # Part 1: C[0, :]
    for j in range(4):
        for k in range(7):
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            add_term(u_idx, v_idx, w_idx)

    # Part 2a: A[1:3, 0:2] * B[0:2, :]
    # Strassen for B[0:2, 0:2]
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        a00 = a_r * m + a_c
        a01 = a_r * m + a_c + 1
        a10 = (a_r + 1) * m + a_c
        a11 = (a_r + 1) * m + a_c + 1
        
        b00 = b_r * p + b_c
        b01 = b_r * p + b_c + 1
        b10 = (b_r + 1) * p + b_c
        b11 = (b_r + 1) * p + b_c + 1
        
        c00 = c_c * n + c_r
        c01 = (c_c + 1) * n + c_r
        c10 = c_c * n + (c_r + 1)
        c11 = (c_c + 1) * n + (c_r + 1)
        
        # M1
        add_terms({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # M2
        add_terms({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
        # M3
        add_terms({a00: 1}, {b01: 1, b11: -1}, {c01: 1})
        # M4
        add_terms({a11: 1}, {b10: 1, b00: -1}, {c10: 1, c00: 1})
        # M5
        add_terms({a00: 1, a01: 1}, {b11: 1}, {c11: -1, c01: 1})
        # M6
        add_terms({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        # M7
        add_terms({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1, c11: 1})

    add_strassen_block(1, 0, 0, 0, 1, 0)
    add_strassen_block(1, 0, 0, 2, 1, 2)
    add_strassen_block(1, 0, 0, 4, 1, 4)
    
    # 2x1 part: A[1:3, 0:2] * B[0:2, 6:7]
    # Term 1: A[:,0] * B[0,6]
    # U: A10(1), A20(1)
    # V: B06(1)
    # W: C16(1), C26(1)
    add_terms({1*m+0: 1, 2*m+0: 1}, {0*p+6: 1}, {6*n+1: 1, 6*n+2: 1})
    # Term 2: A[:,1] * B[1,6]
    # U: A11(1), A21(1)
    # V: B16(1)
    # W: C16(1), C26(1)
    add_terms({1*m+1: 1, 2*m+1: 1}, {1*p+6: 1}, {6*n+1: 1, 6*n+2: 1})

    # Part 2b: A[1:3, 2:4] * B[2:4, :]
    add_strassen_block(1, 2, 2, 0, 1, 0)
    add_strassen_block(1, 2, 2, 2, 1, 2)
    add_strassen_block(1, 2, 2, 4, 1, 4)
    
    # 2x1 part: A[1:3, 2:4] * B[2:4, 6:7]
    # Term 1: A[:,2] * B[2,6]
    add_terms({1*m+2: 1, 2*m+2: 1}, {2*p+6: 1}, {6*n+1: 1, 6*n+2: 1})
    # Term 2: A[:,3] * B[3,6]
    add_terms({1*m+3: 1, 2*m+3: 1}, {3*p+6: 1}, {6*n+1: 1, 6*n+2: 1})

    return {"U": U, "V": V, "W": W}
