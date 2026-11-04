import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_u(i, j, c, val):
        U[i * m + j, c] += val
    def add_v(j, k, c, val):
        V[j * p + k, c] += val
    def add_w(k, i, c, val):
        W[k * n + i, c] += val

    # Strassen 2x2 multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)(b11)
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 - m5 + m7 + m4
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # a_idx: [a00, a01, a10, a11] -> (i,j) pairs
    # b_idx: [b00, b01, b10, b11] -> (j,k) pairs
    # c_idx: [C00, C01, C10, C11] -> (k,i) pairs
    
    def add_strassen(a_idx, b_idx, c_idx, offset):
        # m1: (a00+a11)(b00+b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+0, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+0, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+0, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+0, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+0, 1)   # C00
        add_w(c_idx[3][0], c_idx[3][1], offset+0, 1)   # C11
        
        # m2: (a10+a11)(b00)
        add_u(a_idx[2][0], a_idx[2][1], offset+1, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+1, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+1, 1)
        add_w(c_idx[2][0], c_idx[2][1], offset+1, 1)   # C10
        add_w(c_idx[3][0], c_idx[3][1], offset+1, -1)  # C11
        
        # m3: a00(b01-b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+2, 1)
        add_v(b_idx[1][0], b_idx[1][1], offset+2, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+2, -1)
        add_w(c_idx[1][0], c_idx[1][1], offset+2, 1)   # C01
        add_w(c_idx[3][0], c_idx[3][1], offset+2, 1)   # C11
        
        # m4: a11(b10-b00)
        add_u(a_idx[3][0], a_idx[3][1], offset+3, 1)
        add_v(b_idx[2][0], b_idx[2][1], offset+3, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+3, -1)
        add_w(c_idx[0][0], c_idx[0][1], offset+3, 1)   # C00
        add_w(c_idx[2][0], c_idx[2][1], offset+3, 1)   # C10
        
        # m5: (a00+a01)(b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+4, 1)
        add_u(a_idx[1][0], a_idx[1][1], offset+4, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+4, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+4, -1)  # C00
        add_w(c_idx[1][0], c_idx[1][1], offset+4, 1)   # C01
        
        # m6: (a10-a00)(b00+b10)
        add_u(a_idx[2][0], a_idx[2][1], offset+5, 1)
        add_u(a_idx[0][0], a_idx[0][1], offset+5, -1)
        add_v(b_idx[0][0], b_idx[0][1], offset+5, 1)
        add_v(b_idx[2][0], b_idx[2][1], offset+5, 1)
        add_w(c_idx[3][0], c_idx[3][1], offset+5, 1)   # C11
        
        # m7: (a01-a11)(b10+b11)
        add_u(a_idx[1][0], a_idx[1][1], offset+6, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+6, -1)
        add_v(b_idx[2][0], b_idx[2][1], offset+6, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+6, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+6, 1)   # C00

    # Block 1: A1 (cols 0,1) * B1a (rows 0,1, cols 0,1)
    # A1: i in {0,1}, j in {0,1}
    # B1a: j in {0,1}, k in {0,1}
    # C1a: k in {0,1}, i in {0,1}
    a_idx1 = [(0,0), (0,1), (1,0), (1,1)]
    b_idx1 = [(0,0), (0,1), (1,0), (1,1)]
    c_idx1 = [(0,0), (1,0), (0,1), (1,1)] # k,i pairs: C00(k=0,i=0), C01(k=1,i=0), C10(k=0,i=1), C11(k=1,i=1)
    # Wait, c_idx order in Strassen is C00, C01, C10, C11.
    # C00: i=0, k=0 -> (0,0)
    # C01: i=0, k=1 -> (1,0)
    # C10: i=1, k=0 -> (0,1)
    # C11: i=1, k=1 -> (1,1)
    # My c_idx1 above: [(0,0), (1,0), (0,1), (1,1)] matches (k,i).
    add_strassen(a_idx1, b_idx1, c_idx1, 0)
    
    # Block 2: A1 (cols 0,1) * B1b (rows 0,1, cols 2,3)
    # B1b: j in {0,1}, k in {2,3}
    # C1b: k in {2,3}, i in {0,1}
    b_idx2 = [(0,2), (0,3), (1,2), (1,3)]
    c_idx2 = [(2,0), (3,0), (2,1), (3,1)] # k,i pairs
    add_strassen(a_idx1, b_idx2, c_idx2, 7)
    
    # Block 3: A2 (col 2) * B2 (row 2, cols 0-3)
    # A2: i in {0,1}, j=2
    # B2: j=2, k in {0,1,2,3}
    # C2: k in {0,1,2,3}, i in {0,1}
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
