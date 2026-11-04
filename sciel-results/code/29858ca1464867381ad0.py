import time

DEADLINE = 115.0

def construct():
    start = time.time()
    
    n, m, p = 3, 3, 8
    
    def A_idx(i, j):
        return i * 3 + j
    
    def B_idx(j, k):
        return j * 3 + k
    
    def W_idx(k, i):
        return k * 3 + i
    
    terms = []
    
    def add_term(u_factors, v_factors, w_factors):
        terms.append((u_factors, v_factors, w_factors))
    
    # Laderman's rank-23 decomposition for 3x3x3.
    # Using a verified formulation. The 23 bilinear forms:
    
    # I'll use the standard Laderman decomposition. Here are the 23 products
    # and the resulting C entries.
    
    # Let me use a known correct implementation pattern.
    
    # For 3x3x3 at column offset co:
    def laderman_block(co):
        # A(i,j): U row = i*3+j
        # B(j,k): V row = j*3+(k+co)
        # C(k,i): W row = (k+co)*3+i
        
        def a(i, j):
            return (A_idx(i, j), 1)
        
        def b(j, k):
            return (B_idx(j, k + co), 1)
        
        def c(k, i):
            return (W_idx(k + co, i), 1)
        
        # 23 Laderman terms
        # Using the formulation where we compute:
        # s1 = a00+a11+a22, s2 = a00+a11, s3 = a00+a22, s4 = a11+a22
        # t1 = b00+b11+b22, t2 = b00+b11, t3 = b00+b22, t4 = b11+b22
        
        # m1 = s1*t1
        add_term([a(0,0), a(1,1), a(2,2)], [b(0,0), b(1,1), b(2,2)], [])
        # m2 = s2*t2
        add_term([a(0,0), a(1,1)], [b(0,0), b(1,1)], [])
        # m3 = s3*t3
        add_term([a(0,0), a(2,2)], [b(0,0), b(2,2)], [])
        # m4 = s4*t4
        add_term([a(1,1), a(2,2)], [b(1,1), b(2,2)], [])
        # m5 = s1*t2
        add_term([a(0,0), a(1,1), a(2,2)], [b(0,0), b(1,1)], [])
        # m6 = s1*t3
        add_term([a(0,0), a(1,1), a(2,2)], [b(0,0), b(2,2)], [])
        # m7 = s1*t4
        add_term([a(0,0), a(1,1), a(2,2)], [b(1,1), b(2,2)], [])
        # m8 = s2*t1
        add_term([a(0,0), a(1,1)], [b(0,0), b(1,1), b(2,2)], [])
        # m9 = s3*t1
        add_term([a(0,0), a(2,2)], [b(0,0), b(1,1), b(2,2)], [])
        # m10 = s4*t1
        add_term([a(1,1), a(2,2)], [b(0,0), b(1,1), b(2,2)], [])
        # m11 = a02*b20
        add_term([a(0,2)], [b(2,0)], [])
        # m12 = a20*b02
        add_term([a(2,0)], [b(0,2)], [])
        # m13 = a12*b21
        add_term([a(1,2)], [b(2,1)], [])
        # m14 = a21*b12
        add_term([a(2,1)], [b(1,2)], [])
        # m15 = a01*b10
        add_term([a(0,1)], [b(1,0)], [])
        # m16 = a10*b01
        add_term([a(1,0)], [b(0,1)], [])
        # m17 = s1*(b00-b11)
        add_term([a(0,0), a(1,1), a(2,2)], [b(0,0), (B_idx(1, 0+co), -1)], [])
        # m18 = (a00-a11+a22)*(b00+b11-b22)
        add_term([a(0,0), (A_idx(1,1), -1), a(2,2)], [b(0,0), b(1,1), (B_idx(2, 2+co), -1)], [])
        # m19 = (a00+a11-a22)*(b00-b11+b22)
        add_term([a(0,0), a(1,1), (A_idx(2,2), -1)], [b(0,0), (B_idx(1, 0+co), -1), b(2,2)], [])
        # m20 = (a00-a11-a22)*(b00+b11+b22)
        add_term([a(0,0), (A_idx(1,1), -1), (A_idx(2,2), -1)], [b(0,0), b(1,1), b(2,2)], [])
        # m21 = (a00+a11)*(b00-b11)
        add_term([a(0,0), a(1,1)], [b(0,0), (B_idx(1, 0+co), -1)], [])
        # m22 = (a00-a11)*(b00+b11)
        add_term([a(0,0), (A_idx(1,1), -1)], [b(0,0), b(1,1)], [])
        # m23 = s1*t1 (duplicate of m1) - this is wrong
        
        # I realize I can't get Laderman right from memory. Let me just use schoolbook.
        # Clear the added terms for this block and use schoolbook.
    
    # Given the difficulty, let me just return the schoolbook rank-72 solution.
    terms.clear()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_factors = [(A_idx(i, j), 1)]
                v_factors = [(B_idx(j, k), 1)]
                w_factors = [(W_idx(k, i), 1)]
                add_term(u_factors, v_factors, w_factors)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_factors, v_factors, w_factors) in enumerate(terms):
        for (row, coeff) in u_factors:
            U[row][col] = coeff
        for (row, coeff) in v_factors:
            V[row][col] = coeff
        for (row, coeff) in w_factors:
            W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
