"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm decomposition
    # r=0: P1 = (a+d)(e+h) -> c00, c01, c10, c11
    # U: a(0,0), d(1,1)
    # V: e(0,0), h(1,1)
    # W: c00(0,0), c01(0,1), c10(1,0), c11(1,1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1
    
    # r=1: P2 = (c+d)e -> c10
    # U: c(1,0), d(1,1)
    # V: e(0,0)
    # W: c10(1,0)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # r=2: P3 = a(f-h) -> c01, c11
    # U: a(0,0)
    # V: f(0,1), h(1,1)
    # W: c01(0,1), c11(1,1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3: P4 = d(g-e) -> c00, c10
    # U: d(1,1)
    # V: g(1,0), e(0,0)
    # W: c00(0,0), c10(1,0)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # r=4: P5 = (a+b)h -> -c00, c01
    # U: a(0,0), b(0,1)
    # V: h(1,1)
    # W: -c00(0,0), c01(0,1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=5: P6 = (c-d)(f+h) -> c11
    # U: c(1,0), d(1,1)
    # V: f(0,1), h(1,1)
    # W: c11(1,1)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[3][5] = 1
    
    # r=6: P7 = (b-c)(e+g) -> c00
    # U: b(0,1), c(1,0)
    # V: e(0,0), g(1,0)
    # W: c00(0,0)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
