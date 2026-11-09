import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    rank = 12
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(alpha, beta, gamma):
        nonlocal col
        # alpha: dict { (i, j): val }
        # beta: dict { (j, k): val }
        # gamma: dict { (i, k): val }
        
        for (i, j), val in alpha.items():
            if val != 0:
                U[i * m + j, col] += val
        for (j, k), val in beta.items():
            if val != 0:
                V[j * p + k, col] += val
        for (i, k), val in gamma.items():
            if val != 0:
                W[k * n + i, col] += val
        col += 1

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Local indices i,j,k in 0..1 map to global i,j,k in 0..1
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    
    # M2 = (A10 + A11)B00 -> C10, C11(-)
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})
    
    # M3 = A00(B10 - B11) -> C01, C11
    add_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1})
    
    # M4 = A11(B01 - B00) -> C00(-), C10
    add_term({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):-1, (1,0):1})
    
    # M5 = (A00 + A01)B11 -> C00(-), C01
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    
    # M7 = (A01 - A11)(B10 + B11) -> C00, C01(-)
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1})

    # 2. Trivial for A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # B col 2, rows 0,1. C col 2, rows 0,1.
    
    # T1: A00 B02 -> C02
    add_term({(0,0):1}, {(0,2):1}, {(0,2):1})
    
    # T2: A01 B12 -> C02
    add_term({(0,1):1}, {(1,2):1}, {(0,2):1})
    
    # T3: A10 B02 -> C12
    add_term({(1,0):1}, {(0,2):1}, {(1,2):1})
    
    # T4: A11 B12 -> C12
    add_term({(1,1):1}, {(1,2):1}, {(1,2):1})

    # 3. Trivial for A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
    # A col 2, rows 0,1. B row 2, cols 0,1,2. C all.
    
    # Rank 1 term
    add_term({(0,2):1, (1,2):1}, {(2,0):1, (2,1):1, (2,2):1}, 
             {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1})

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
