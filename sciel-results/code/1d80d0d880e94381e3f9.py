"""Strassen-based rank-21 decomposition for 2x2 x 2x6 matrix multiplication."""
DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    # A: n x m = 2x2, B: m x p = 2x6, C: p x n = 6x2
    # Partition B into 3 blocks of width 2 along k
    # Each block: 2x2 x 2x2 = rank 7 via Strassen
    # Total rank = 21
    
    rank = 21
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 21
    V = [[0.0] * rank for _ in range(m * p)]   # 12 x 21
    W = [[0.0] * rank for _ in range(p * n)]   # 12 x 21
    
    # Strassen terms for 2x2 x 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1=(a+d)(e+h), P2=(c+d)e, P3=a(f-h), P4=d(g-e)
    # P5=(a+b)h, P6=(c-a)(e+g), P7=(b-d)(h-g)
    # C[0,0]=P1+P4+P5+P7, C[0,1]=P3+P5
    # C[1,0]=P2+P4, C[1,1]=P1-P2+P3+P6
    
    # For each term, define coefficients for A entries (a,b,c,d) at indices 0,1,2,3
    # and B entries (e,f,g,h) at indices 0,1,2,3
    # A indices: a=0, b=1, c=2, d=3
    # B local indices: e=0, f=1, g=2, h=3
    # C local indices: C00=0, C01=1, C10=2, C11=3
    
    # Each Strassen term: (u_coeffs on A, v_coeffs on B, w_coeffs on C)
    # P1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C00=1, C01=0, C10=0, C11=1
    # P2 = (c+d)e: U: c=1,d=1; V: e=1; W: C10=1
    # P3 = a(f-h): U: a=1; V: f=1,h=-1; W: C01=1, C11=1
    # P4 = d(g-e): U: d=1; V: g=1,e=-1; W: C00=1, C10=1
    # P5 = (a+b)h: U: a=1,b=1; V: h=1; W: C00=1, C01=1
    # P6 = (c-a)(e+g): U: c=1,a=-1; V: e=1,g=1; W: C11=1
    # P7 = (b-d)(h-g): U: b=1,d=-1; V: h=1,g=-1; W: C00=1
    
    strassen_terms = [
        # (u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_C00, w_C01, w_C10, w_C11)
        (1, 0, 0, 1,  1, 0, 0, 1,  1, 0, 0, 1),   # P1
        (0, 0, 1, 1,  1, 0, 0, 0,  0, 0, 1, 0),   # P2
        (1, 0, 0, 0,  0, 1, 0,-1,  0, 1, 0, 1),   # P3
        (0, 0, 0, 1, -1, 0, 1, 0,  1, 0, 1, 0),   # P4
        (1, 1, 0, 0,  0, 0, 0, 1,  1, 1, 0, 0),   # P5
        (1, 0, 1, 0,  1, 0, 1, 0,  0, 0, 0, 1),   # P6: (c-a)(e+g) -> U: a=-1,c=1
        (0, 1, 0,-1,  0, 0, 1, 1,  1, 0, 0, 0),   # P7: (b-d)(h-g) -> U: b=1,d=-1; V: g=-1,h=1
    ]
    
    # Fix P6: U should be a=-1, c=1
    strassen_terms[5] = (-1, 0, 1, 0,  1, 0, 1, 0,  0, 0, 0, 1)
    # Fix P7: U: b=1, d=-1; V: g=-1, h=1; W: C00=1
    strassen_terms[6] = (0, 1, 0, -1,  0, 0, -1, 1,  1, 0, 0, 0)
    
    for block in range(3):
        # B block starts at column 2*block in the k dimension
        # B global indices: B[j*p + k] where j=0,1 and k=2*block, 2*block+1
        # e = B[0, 2*block] = V index 0*6 + 2*block = 2*block
        # f = B[0, 2*block+1] = V index 2*block + 1
        # g = B[1, 2*block] = V index 6 + 2*block
        # h = B[1, 2*block+1] = V index 6 + 2*block + 1
        
        # C global indices: C[k*n + i] where k=2*block, 2*block+1 and i=0,1
        # C00 = C[2*block*2 + 0] = W index 4*block
        # C01 = C[2*block*2 + 1] = W index 4*block + 1
        # C10 = C[(2*block+1)*2 + 0] = W index 4*block + 2
        # C11 = C[(2*block+1)*2 + 1] = W index 4*block + 3
        
        for s in range(7):
            col = 7 * block + s
            u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_C00, w_C01, w_C10, w_C11 = strassen_terms[s]
            
            # A indices: a=0, b=1, c=2, d=3 (i*m+j)
            U[0][col] = u_a
            U[1][col] = u_b
            U[2][col] = u_c
            U[3][col] = u_d
            
            # V indices for this block
            V[2 * block][col] = v_e
            V[2 * block + 1][col] = v_f
            V[6 + 2 * block][col] = v_g
            V[6 + 2 * block + 1][col] = v_h
            
            # W indices for this block
            W[4 * block][col] = w_C00
            W[4 * block + 1][col] = w_C01
            W[4 * block + 2][col] = w_C10
            W[4 * block + 3][col] = w_C11
    
    return {"U": U, "V": V, "W": W}
