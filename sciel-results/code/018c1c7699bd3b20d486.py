import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_block(u_a00, u_a01, u_a10, u_a11, v_b00, v_b01, v_b10, v_b11, w_c00, w_c01, w_c10, w_c11):
        nonlocal col
        terms = [
            ({u_a00:1, u_a11:1}, {v_b00:1, v_b11:1}, {w_c00:1, w_c11:1}),
            ({u_a10:1, u_a11:1}, {v_b00:1}, {w_c10:1}),
            ({u_a00:1}, {v_b01:1, v_b11:-1}, {w_c01:1}),
            ({u_a11:1}, {v_b10:1, v_b00:-1}, {w_c10:1, w_c00:-1}),
            ({u_a00:1, u_a01:1}, {v_b11:1}, {w_c01:1, w_c00:-1}),
            ({u_a10:1, u_a00:-1}, {v_b00:1, v_b01:1}, {w_c11:1}),
            ({u_a01:1, u_a11:-1}, {v_b10:1, v_b11:1}, {w_c00:1, w_c11:-1})
        ]
        for u_c, v_c, w_c in terms:
            for idx, val in u_c.items():
                U[idx][col] += val
            for idx, val in v_c.items():
                V[idx][col] += val
            for idx, val in w_c.items():
                W[idx][col] += val
            col += 1

    def add_schoolbook_block(u_a00, u_a01, u_a10, u_a11, v_b0, v_b1, w_c0, w_c1):
        nonlocal col
        terms = [
            ({u_a00:1}, {v_b0:1}, {w_c0:1}),
            ({u_a01:1}, {v_b1:1}, {w_c0:1}),
            ({u_a10:1}, {v_b0:1}, {w_c1:1}),
            ({u_a11:1}, {v_b1:1}, {w_c1:1})
        ]
        for u_c, v_c, w_c in terms:
            for idx, val in u_c.items():
                U[idx][col] += val
            for idx, val in v_c.items():
                V[idx][col] += val
            for idx, val in w_c.items():
                W[idx][col] += val
            col += 1

    # Top Part: A1 (A[0:2, 0:2]) * Btop (B[0:2, :])
    # U indices for A1:
    # a00: U[0], a01: U[1], a10: U[4], a11: U[5]
    
    # Block 1: Btop cols 0,1
    # V indices: b00=V[0], b01=V[1], b10=V[7], b11=V[8]
    # W indices: c00=W[0], c01=W[2], c10=W[1], c11=W[3]
    add_strassen_block(0, 1, 4, 5, 0, 1, 7, 8, 0, 2, 1, 3)
    
    # Block 2: Btop cols 2,3
    # V indices: b00=V[2], b01=V[3], b10=V[9], b11=V[10]
    # W indices: c00=W[4], c01=W[6], c10=W[5], c11=W[7]
    add_strassen_block(0, 1, 4, 5, 2, 3, 9, 10, 4, 6, 5, 7)
    
    # Block 3: Btop cols 4,5
    # V indices: b00=V[4], b01=V[5], b10=V[11], b11=V[12]
    # W indices: c00=W[8], c01=W[10], c10=W[9], c11=W[11]
    add_strassen_block(0, 1, 4, 5, 4, 5, 11, 12, 8, 10, 9, 11)
    
    # Block 4: Btop col 6
    # V indices: b0=V[6], b1=V[13]
    # W indices: c0=W[12], c1=W[13]
    add_schoolbook_block(0, 1, 4, 5, 6, 13, 12, 13)
    
    # Bottom Part: A2 (A[0:2, 2:4]) * Bbottom (B[2:4, :])
    # U indices for A2:
    # a00: U[2], a01: U[3], a10: U[6], a11: U[7]
    
    # Block 5: Bbottom cols 0,1
    # V indices: b00=V[14], b01=V[15], b10=V[21], b11=V[22]
    # W indices: c00=W[0], c01=W[2], c10=W[1], c11=W[3]
    add_strassen_block(2, 3, 6, 7, 14, 15, 21, 22, 0, 2, 1, 3)
    
    # Block 6: Bbottom cols 2,3
    # V indices: b00=V[16], b01=V[17], b10=V[23], b11=V[24]
    # W indices: c00=W[4], c01=W[6], c10=W[5], c11=W[7]
    add_strassen_block(2, 3, 6, 7, 16, 17, 23, 24, 4, 6, 5, 7)
    
    # Block 7: Bbottom cols 4,5
    # V indices: b00=V[18], b01=V[19], b10=V[25], b11=V[26]
    # W indices: c00=W[8], c01=W[10], c10=W[9], c11=W[11]
    add_strassen_block(2, 3, 6, 7, 18, 19, 25, 26, 8, 10, 9, 11)
    
    # Block 8: Bbottom col 6
    # V indices: b0=V[20], b1=V[27]
    # W indices: c0=W[12], c1=W[13]
    add_schoolbook_block(2, 3, 6, 7, 20, 27, 12, 13)
    
    return {"U": U, "V": V, "W": W}
