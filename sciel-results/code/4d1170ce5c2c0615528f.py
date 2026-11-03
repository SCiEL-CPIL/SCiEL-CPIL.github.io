"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 8 blocks * 2 * 7 (Winograd) + 2 blocks * 2 * 4 (Schoolbook 2x2x1) = 112 + 16 = 128
    rank = 128
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Winograd 2x2x2 decomposition
    # local_a_rows: [r1, r2] global rows in A
    # local_a_cols: [c1, c2] global cols in A
    # local_b_rows: [r1, r2] global rows in B
    # local_b_cols: [c1, c2] global cols in B
    # local_c_rows: [r1, r2] global cols in C (which are rows in W)
    # local_c_cols: [c1, c2] global rows in C (which are cols in W)
    def add_winograd(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2):
        nonlocal col_idx
        # Winograd coefficients for 2x2x2
        # m1 = (a+d)(e+h)
        # m2 = (c+d)(e+g)
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a-b)(e+f)
        # m6 = (b-c)(g+h)
        # m7 = (b+d)(f-h)
        
        # Local indices in 2x2 block:
        # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        # C: (0,0)=C00, (0,1)=C01, (1,0)=C10, (1,1)=C11
        
        # Global indices mapping:
        # A index: r * m + c
        # B index: r * p + c
        # C index: r * n + c
        
        # Term 1: m1
        U[a_r1 * m + a_c1][col_idx] += 1
        U[a_r2 * m + a_c2][col_idx] += 1
        V[b_r1 * p + b_c1][col_idx] += 1
        V[b_r2 * p + b_c2][col_idx] += 1
        W[c_r1 * n + c_c1][col_idx] += 1   # C00
        W[c_r1 * n + c_c2][col_idx] += 0   # C01
        W[c_r2 * n + c_c1][col_idx] += 1   # C10
        W[c_r2 * n + c_c2][col_idx] += -1  # C11
        col_idx += 1
        
        # Term 2: m2
        U[a_r1 * m + a_c2][col_idx] += 0 # c is (1,0) -> a_r2, a_c1? No.
        # Let's stick to explicit variables to avoid confusion.
        # A: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
        # B: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
        
        # m2 = (c+d)(e+g)
        # c is A[1,0] -> a_r2, a_c1
        # d is A[1,1] -> a_r2, a_c2
        # e is B[0,0] -> b_r1, b_c1
        # g is B[1,0] -> b_r2, b_c1
        U[a_r2 * m + a_c1][col_idx] += 1
        U[a_r2 * m + a_c2][col_idx] += 1
        V[b_r1 * p + b_c1][col_idx] += 1
        V[b_r2 * p + b_c1][col_idx] += 1
        W[c_r1 * n + c_c1][col_idx] += 0   # C00
        W[c_r1 * n + c_c2][col_idx] += 0   # C01
        W[c_r2 * n + c_c1][col_idx] += 1   # C10
        W[c_r2 * n + c_c2][col_idx] += 1   # C11
        col_idx += 1
        
        # m3 = a(f-h)
        # a is A[0,0] -> a_r1, a_c1
        # f is B[0,1] -> b_r1, b_c2
        # h is B[1,1] -> b_r2, b_c2
        U[a_r1 * m + a_c1][col_idx] += 1
        V[b_r1 * p + b_c2][col_idx] += 1
        V[b_r2 * p + b_c2][col_idx] += -1
        W[c_r1 * n + c_c1][col_idx] += 1   # C00
        W[c_r1 * n + c_c2][col_idx] += 0   # C01
        W[c_r2 * n + c_c1][col_idx] += -1  # C10
        W[c_r2 * n + c_c2][col_idx] += 0   # C11
        col_idx += 1
        
        # m4 = d(g-e)
        # d is A[1,1] -> a_r2, a_c2
        # g is B[1,0] -> b_r2, b_c1
        # e is B[0,0] -> b_r1, b_c1
        U[a_r2 * m + a_c2][col_idx] += 1
        V[b_r2 * p + b_c1][col_idx] += 1
        V[b_r1 * p + b_c1][col_idx] += -1
        W[c_r1 * n + c_c1][col_idx] += -1  # C00
        W[c_r1 * n + c_c2][col_idx] += 1   # C01
        W[c_r2 * n + c_c1][col_idx] += 0   # C10
        W[c_r2 * n + c_c2][col_idx] += 1   # C11
        col_idx += 1
        
        # m5 = (a-b)(e+f)
        # a is A[0,0], b is A[0,1]
        # e is B[0,0], f is B[0,1]
        U[a_r1 * m + a_c1][col_idx] += 1
        U[a_r1 * m + a_c2][col_idx] += -1
        V[b_r1 * p + b_c1][col_idx] += 1
        V[b_r1 * p + b_c2][col_idx] += 1
        W[c_r1 * n + c_c1][col_idx] += 0   # C00
        W[c_r1 * n + c_c2][col_idx] += 1   # C01
        W[c_r2 * n + c_c1][col_idx] += -1  # C10
        W[c_r2 * n + c_c2][col_idx] += 0   # C11
        col_idx += 1
        
        # m6 = (b-c)(g+h)
        # b is A[0,1], c is A[1,0]
        # g is B[1,0], h is B[1,1]
        U[a_r1 * m + a_c2][col_idx] += 1
        U[a_r2 * m + a_c1][col_idx] += -1
        V[b_r2 * p + b_c1][col_idx] += 1
        V[b_r2 * p + b_c2][col_idx] += 1
        W[c_r1 * n + c_c1][col_idx] += 0   # C00
        W[c_r1 * n + c_c2][col_idx] += -1  # C01
        W[c_r2 * n + c_c1][col_idx] += 0   # C10
        W[c_r2 * n + c_c2][col_idx] += 1   # C11
        col_idx += 1
        
        # m7 = (b+d)(f-h)
        # b is A[0,1], d is A[1,1]
        # f is B[0,1], h is B[1,1]
        U[a_r1 * m + a_c2][col_idx] += 1
        U[a_r2 * m + a_c2][col_idx] += 1
        V[b_r1 * p + b_c2][col_idx] += 1
        V[b_r2 * p + b_c2][col_idx] += -1
        W[c_r1 * n + c_c1][col_idx] += -1  # C00
        W[c_r1 * n + c_c2][col_idx] += -1  # C01
        W[c_r2 * n + c_c1][col_idx] += 0   # C10
        W[c_r2 * n + c_c2][col_idx] += 0   # C11
        col_idx += 1

    # Helper to add a Schoolbook 2x2x1 decomposition
    # A is 2x2, B is 2x1, C is 2x1
    def add_schoolbook_2x2x1(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c, c_r, c_c1, c_c2):
        nonlocal col_idx
        # c1 = a*e + b*g
        # c2 = c*e + d*g
        # Term 1: a*e
        U[a_r1 * m + a_c1][col_idx] += 1
        V[b_r1 * p + b_c][col_idx] += 1
        W[c_r * n + c_c1][col_idx] += 1
        col_idx += 1
        
        # Term 2: b*g
        U[a_r1 * m + a_c2][col_idx] += 1
        V[b_r2 * p + b_c][col_idx] += 1
        W[c_r * n + c_c1][col_idx] += 1
        col_idx += 1
        
        # Term 3: c*e
        U[a_r2 * m + a_c1][col_idx] += 1
        V[b_r1 * p + b_c][col_idx] += 1
        W[c_r * n + c_c2][col_idx] += 1
        col_idx += 1
        
        # Term 4: d*g
        U[a_r2 * m + a_c2][col_idx] += 1
        V[b_r2 * p + b_c][col_idx] += 1
        W[c_r * n + c_c2][col_idx] += 1
        col_idx += 1

    # Iterate over row blocks I (0, 1) and col blocks K (0, 1, 2, 3, 4)
    for I in range(2):
        for K in range(4):
            # Inner dim split S=0 (j=0,1) and S=1 (j=2,3)
            for S in range(2):
                # A rows: 2I, 2I+1
                # A cols: 2S, 2S+1
                # B rows: 2S, 2S+1
                # B cols: 2K, 2K+1
                # C cols (rows in W): 2K, 2K+1
                # C rows (cols in W): 2I, 2I+1
                
                a_r1 = 2 * I
                a_r2 = 2 * I + 1
                a_c1 = 2 * S
                a_c2 = 2 * S + 1
                
                b_r1 = 2 * S
                b_r2 = 2 * S + 1
                b_c1 = 2 * K
                b_c2 = 2 * K + 1
                
                c_r1 = 2 * K
                c_r2 = 2 * K + 1
                c_c1 = 2 * I
                c_c2 = 2 * I + 1
                
                add_winograd(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2)
                
        # Handle the last column block K=4 (size 1)
        # C col: 4
        # C rows: 2I, 2I+1
        for S in range(2):
            # A rows: 2I, 2I+1
            # A cols: 2S, 2S+1
            # B rows: 2S, 2S+1
            # B col: 4
            # C col: 4
            # C rows: 2I, 2I+1
            
            a_r1 = 2 * I
            a_r2 = 2 * I + 1
            a_c1 = 2 * S
            a_c2 = 2 * S + 1
            
            b_r1 = 2 * S
            b_r2 = 2 * S + 1
            b_c = 4
            
            c_r = 4
            c_c1 = 2 * I
            c_c2 = 2 * I + 1
            
            add_schoolbook_2x2x1(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c, c_r, c_c1, c_c2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
