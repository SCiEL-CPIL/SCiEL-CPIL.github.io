import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (6, 39)
    # V: (m*p, rank) = (21, 39)
    # W: (p*n, rank) = (14, 39)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx):
        nonlocal col
        for idx, coeff in u_idx:
            U[idx][col] += coeff
        for idx, coeff in v_idx:
            V[idx][col] += coeff
        for idx, coeff in w_idx:
            W[idx][col] += coeff
        col += 1
        
    # Part 1: A_col2 * B_row2
    # A indices: i*m + 2. i in {0,1}.
    # B indices: 2*p + k. k in {0..6}.
    # W indices: k*n + i.
    for i in range(n):
        for k in range(p):
            u_idx = [(i * m + 2, 1)]
            v_idx = [(2 * p + k, 1)]
            w_idx = [(k * n + i, 1)]
            add_term(u_idx, v_idx, w_idx)
            
    # Part 2: A_cols01 * B_rows01
    # A indices for cols 0,1:
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices for rows 0,1:
    # B[0,k] = k, B[1,k] = 7+k
    
    # Split B_rows01 into 3 blocks of 2 cols and 1 block of 1 col
    # Blocks: (0,1), (2,3), (4,5), (6)
    
    for block in range(3):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # B indices
        e = k0       # B[0, k0]
        f = k1       # B[0, k1]
        g = 7 + k0   # B[1, k0]
        h = 7 + k1   # B[1, k1]
        
        # W indices for 2x2 output block at cols k0, k1
        # C[0, k0] -> k0*n + 0 = 2*k0
        # C[0, k1] -> k1*n + 0 = 2*k1
        # C[1, k0] -> k0*n + 1 = 2*k0+1
        # C[1, k1] -> k1*n + 1 = 2*k1+1
        
        w00 = 2 * k0
        w01 = 2 * k1
        w10 = 2 * k0 + 1
        w11 = 2 * k1 + 1
        
        # Strassen 7 terms
        # 1. (a+d)(e+h)
        add_term(
            [(0, 1), (4, 1)],
            [(e, 1), (h, 1)],
            [(w00, 1), (w11, 1)]
        )
        
        # 2. (c+d)e
        add_term(
            [(3, 1), (4, 1)],
            [(e, 1)],
            [(w10, 1), (w11, -1)]
        )
        
        # 3. a(f-h)
        add_term(
            [(0, 1)],
            [(f, 1), (h, -1)],
            [(w01, 1), (w11, 1)]
        )
        
        # 4. d(g-e)
        add_term(
            [(4, 1)],
            [(g, 1), (e, -1)],
            [(w00, 1), (w10, 1)]
        )
        
        # 5. (a+b)h
        add_term(
            [(0, 1), (1, 1)],
            [(h, 1)],
            [(w00, -1), (w01, 1)]
        )
        
        # 6. (c-a)(e+f)
        add_term(
            [(3, 1), (0, -1)],
            [(e, 1), (f, 1)],
            [(w11, 1)]
        )
        
        # 7. (b-d)(g+h)
        add_term(
            [(1, 1), (4, -1)],
            [(g, 1), (h, 1)],
            [(w00, 1)]
        )
        
    # Last block: col 6
    k = 6
    e = k       # B[0, 6]
    g = 7 + k   # B[1, 6]
    
    w00 = 2 * k
    w10 = 2 * k + 1
    
    # 4 terms for 2x2 * 2x1
    # C[0,6] = a*e + b*g
    add_term([(0, 1)], [(e, 1)], [(w00, 1)])
    add_term([(1, 1)], [(g, 1)], [(w00, 1)])
    
    # C[1,6] = c*e + d*g
    add_term([(3, 1)], [(e, 1)], [(w10, 1)])
    add_term([(4, 1)], [(g, 1)], [(w10, 1)])
    
    return {"U": U, "V": V, "W": W}
