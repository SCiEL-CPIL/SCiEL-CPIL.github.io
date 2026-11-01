"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # Using Strassen for 2x2 blocks and Schoolbook for the remainder.
    # 15 = 7 * 2 + 1
    # 7 blocks of 2x2, 1 block of 2x1.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # Strassen decomposition for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b01
    # m3 = a00(b10-b11)
    # m4 = a11(b00-b10)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m2 + m3 - m7 + m5
    # c10 = m2 + m4 - m3 + m6
    # c11 = m1 + m3 - m2 + m6
    
    # Indices for A (n*m = 4): 00, 01, 10, 11
    # Indices for B (m*p = 30): 
    # Block j (0..6) uses B indices:
    # b00: j*2 + 0
    # b01: j*2 + 1
    # b10: j*2 + 2
    # b11: j*2 + 3
    # Wait, B is 2x15.
    # Row 0: B[0,0]..B[0,14]
    # Row 1: B[1,0]..B[1,14]
    # In flattened V (m*p), index is row*m? No, V is (m*p, rank).
    # V index: row * p + col?
    # Task says V shape (m*p, rank).
    # Usually V[j*p + k] corresponds to B[j,k].
    
    # Let's define the 7 Strassen terms for a generic block j
    # A indices: 0, 1, 2, 3 for 00, 01, 10, 11
    # B indices for block j:
    # b00 = 0*15 + (2*j)
    # b01 = 0*15 + (2*j + 1)
    # b10 = 1*15 + (2*j)
    # b11 = 1*15 + (2*j + 1)
    
    for j in range(7):
        b00 = 0 * 15 + (2 * j)
        b01 = 0 * 15 + (2 * j + 1)
        b10 = 1 * 15 + (2 * j)
        b11 = 1 * 15 + (2 * j + 1)
        
        # m1
        u = [1, 0, 0, 1] # a00+a11
        v = [0]*30
        v[b00] = 1
        v[b11] = 1
        w = [0]*30 # W is p*n = 15*2 = 30
        # c00 += m1
        # c00 is at index 0*2 + 0?
        # W[k*n+i]. k is row of C (0..14), i is col of C (0..1).
        # C is 15x2.
        # C[0,0] is k=0, i=0. Index 0.
        # C[0,1] is k=0, i=1. Index 1.
        # C[1,0] is k=1, i=0. Index 2.
        # C[1,1] is k=1, i=1. Index 3.
        # Block j corresponds to C rows 2*j and 2*j+1.
        c00_idx = (2*j) * 2 + 0
        c01_idx = (2*j) * 2 + 1
        c10_idx = (2*j+1) * 2 + 0
        c11_idx = (2*j+1) * 2 + 1
        
        w[c00_idx] += 1
        add_term(u, v, w)
        
        # m2
        u = [0, 0, 1, 1] # a10+a11
        v = [0]*30
        v[b01] = 1
        w = [0]*30
        w[c01_idx] += 1
        w[c10_idx] += 1
        add_term(u, v, w)
        
        # m3
        u = [1, 0, 0, 0] # a00
        v = [0]*30
        v[b10] = 1
        v[b11] = -1
        w = [0]*30
        w[c01_idx] += 1
        w[c10_idx] -= 1
        add_term(u, v, w)
        
        # m4
        u = [0, 0, 0, 1] # a11
        v = [0]*30
        v[b00] = 1
        v[b10] = -1
        w = [0]*30
        w[c00_idx] += 1
        w[c10_idx] += 1
        add_term(u, v, w)
        
        # m5
        u = [1, 1, 0, 0] # a00+a01
        v = [0]*30
        v[b11] = 1
        w = [0]*30
        w[c00_idx] -= 1
        w[c01_idx] += 1
        add_term(u, v, w)
        
        # m6
        u = [0, 0, 1, -1] # a10-a00
        v = [0]*30
        v[b00] = 1
        v[b10] = 1
        w = [0]*30
        w[c10_idx] += 1
        w[c11_idx] += 1
        add_term(u, v, w)
        
        # m7
        u = [0, 1, 0, -1] # a01-a11
        v = [0]*30
        v[b01] = 1
        v[b11] = 1
        w = [0]*30
        w[c00_idx] += 1
        w[c01_idx] -= 1
        w[c11_idx] += 1
        add_term(u, v, w)

    # Last block: 2x1
    # B indices:
    # b00 = 0*15 + 14
    # b10 = 1*15 + 14
    b00 = 0 * 15 + 14
    b10 = 1 * 15 + 14
    
    # C indices:
    # c00 = 14*2 + 0 = 28
    # c10 = 14*2 + 1 = 29? No.
    # C is 15x2.
    # Row 14, Col 0: 14*2 + 0 = 28.
    # Row 14, Col 1: 14*2 + 1 = 29.
    
    c00 = 14 * 2 + 0
    c10 = 14 * 2 + 1 # Wait. C[k, i]. k=14, i=0 and i=1.
    
    # A00 * B00
    u = [1, 0, 0, 0]
    v = [0]*30
    v[b00] = 1
    w = [0]*30
    w[c00] += 1
    add_term(u, v, w)
    
    # A01 * B10
    u = [0, 1, 0, 0]
    v = [0]*30
    v[b10] = 1
    w = [0]*30
    w[c00] += 1
    add_term(u, v, w)
    
    # A10 * B00
    u = [0, 0, 1, 0]
    v = [0]*30
    v[b00] = 1
    w = [0]*30
    w[c10] += 1
    add_term(u, v, w)
    
    # A11 * B10
    u = [0, 0, 0, 1]
    v = [0]*30
    v[b10] = 1
    w = [0]*30
    w[c10] += 1
    add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
