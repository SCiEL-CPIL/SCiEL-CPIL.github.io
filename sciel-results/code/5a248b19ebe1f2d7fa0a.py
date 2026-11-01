import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = (a-b)(g+h)
    # P4 = d(f-g)
    # P5 = (a+e)f
    # P6 = (c-b)(h-f)
    # P7 = (a+c)g
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P6
    
    # Indices for U (A): 0:a, 1:b, 2:c, 3:d
    # Indices for V (B): 0:e, 1:f, 2:g, 3:h (for a single 2x2 block)
    # Indices for W (C): 0:C11, 1:C12, 2:C21, 3:C22 (for a single 2x2 block)
    
    # Strassen terms (u, v, w)
    # u: coefficients for a,b,c,d
    # v: coefficients for e,f,g,h
    # w: coefficients for C11,C12,C21,C22
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1: (a+d)(e+h) -> C11(1), C22(1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2: (c+d)e -> C21(1), C22(-1)
        ([1, -1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]), # P3: (a-b)(g+h) -> C12(1)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]), # P4: d(f-g) -> C11(1), C21(1)
        ([1, 0, 0, 0], [0, 1, 0, 0], [-1, 1, 0, 0]), # P5: a*f -> C11(-1), C12(1)  (Note: P5 in Strassen is often (a+e)f, but here we split? No, rank 7 requires specific forms)
        # Actually, P5 in the standard set is (a+e)f which is invalid.
        # The valid rank 7 set uses:
        # P5 = (a+e)f is NOT valid.
        # The valid set is:
        # P1=(a+d)(e+h), P2=(c+d)e, P3=(a-b)(g+h), P4=d(f-g), P5=(a+e)f...
        # I will use the 32 rank solution to ensure correctness.
    ]
    
    # Fallback to 32
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
