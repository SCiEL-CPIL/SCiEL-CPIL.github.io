import time

DIMENSIONS = [2, 3, 10]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    DEADLINE = start + 120  # safety margin; construction is trivial
    
    terms = []  # list of (u, v, w) dicts
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Split B's p columns into 5 blocks of 2 columns.
    # For each block t (cols c0=2t, c1=2t+1):
    #   C_t (2x2) = A (2x3) * B_t (3x2)
    #   Split A = [A_L (2x2, cols 0,1), A_R (2x1, col 2)]
    #   Split B_t = [B_tL (2x2, rows 0,1), B_tR (1x2, row 2)]
    #   C_t = A_L * B_tL + A_R * B_tR
    #   A_L*B_tL: Strassen (7 terms)
    #   A_R*B_tR: outer product (4 terms)
    for t in range(5):
        c0 = 2*t
        c1 = 2*t + 1
        # Strassen on A_L (2x2) * B_tL (2x2)
        # a(i,j)=A[i,j] for i,j in {0,1}; b(j,i)=B[j, c0+i] for j in {0,1}, i in {0,1}
        # M1
        add_term({(0,0):1,(1,1):1}, {(0,c0):1,(1,c1):1}, {(c0,0):1,(c1,1):1})
        # M2
        add_term({(1,0):1,(1,1):1}, {(0,c0):1}, {(c0,1):1})
        # M3
        add_term({(0,0):1}, {(0,c1):1,(1,c1):-1}, {(c1,0):1,(c1,1):-1})
        # M4
        add_term({(1,1):1}, {(1,c0):1,(0,c0):-1}, {(c0,1):1,(c0,0):-1})
        # M5
        add_term({(0,0):1,(0,1):1}, {(1,c1):1}, {(c1,0):1,(c1,1):1})
        # M6
        add_term({(1,0):1,(0,0):-1}, {(0,c0):1,(0,c1):1}, {(c0,0):1})
        # M7
        add_term({(0,1):1,(1,1):-1}, {(1,c0):1,(1,c1):1}, {(c0,0):1,(c1,1):1})
        # Outer product A_R * B_tR (4 terms)
        add_term({(0,2):1}, {(2,c0):1}, {(c0,0):1})
        add_term({(0,2):1}, {(2,c1):1}, {(c1,0):1})
        add_term({(1,2):1}, {(2,c0):1}, {(c0,1):1})
        add_term({(1,2):1}, {(2,c1):1}, {(c1,1):1})
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c, (u, v, w) in enumerate(terms):
        for (i,j), coef in u.items():
            U[i*m + j][c] = coef
        for (j,k), coef in v.items():
            V[j*p + k][c] = coef
        for (k,i), coef in w.items():
            W[k*n + i][c] = coef
    return {"U": U, "V": V, "W": W}
