import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2 columns
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # --- Strassen Part (Rank 7) ---
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {k0, k1}
        # C indices: i in {0,1}, k in {k0, k1}
        
        # M1 = (a11 + a22)(b11 + b22)
        # U: a11 (idx 0), a22 (idx 3)
        # V: b11 (idx 0*p + k0? No, V index is j*p + k)
        # j=0, k=k0 -> 0*10+k0 = k0
        # j=1, k=k0 -> 1*10+k0 = 10+k0
        # j=0, k=k1 -> k1
        # j=1, k=k1 -> 10+k1
        
        # Let's define helper to add term
        def add_term(u_rows, v_rows, w_rows, coeff_u, coeff_v, coeff_w):
            nonlocal col
            for r_u in u_rows:
                U[r_u][col] += coeff_u
            for r_v in v_rows:
                V[r_v][col] += coeff_v
            for r_w in w_rows:
                W[r_w][col] += coeff_w
            col += 1

        # M1
        # U: a11 (0), a22 (3) with +1
        # V: b11 (k0), b22 (10+k0) with +1
        # W: c11 (k0*2+0), c12 (k1*2+0), c21 (k0*2+1), c22 (k1*2+1)
        # C11 = M1... C12 = ... C21 = ... C22 = ...
        # We need to distribute M1 to W.
        # W index: k*n + i.
        # c11: k0*2 + 0
        # c12: k1*2 + 0
        # c21: k0*2 + 1
        # c22: k1*2 + 1
        
        # M1 contributes to:
        # C11: +1
        # C12: 0
        # C21: 0
        # C22: +1
        
        add_term([0, 3], [k0, 10+k0], [k0*2+0, k1*2+1], [1, 1], [1, 1], [1, 1])
        
        # M2 = (a21 + a22)b11
        # U: a21 (1), a22 (3)
        # V: b11 (k0)
        # W: C21 (+1), C22 (-1)
        add_term([1, 3], [k0], [k0*2+1, k1*2+1], [1, 1], [1], [1, -1])
        
        # M3 = a11(b12 - b22)
        # U: a11 (0)
        # V: b12 (k1), b22 (10+k1)
        # W: C12 (+1), C22 (+1)
        add_term([0], [k1, 10+k1], [k1*2+0, k1*2+1], [1], [1, -1], [1, 1])
        
        # M4 = a22(b21 - b11)
        # U: a22 (3)
        # V: b21 (10+k0), b11 (k0)
        # W: C11 (+1)
        add_term([3], [10+k0, k0], [k0*2+0], [1], [1, -1], [1])
        
        # M5 = (a11 + a12)b22
        # U: a11 (0), a12 (2)
        # V: b22 (10+k0)
        # W: C11 (-1), C12 (+1)
        add_term([0, 2], [10+k0], [k0*2+0, k1*2+0], [1, 1], [1], [-1, 1])
        
        # M6 = (a21 - a11)(b11 + b12)
        # U: a21 (1), a11 (0)
        # V: b11 (k0), b12 (k1)
        # W: C22 (+1)
        add_term([1, 0], [k0, k1], [k1*2+1], [1, -1], [1, 1], [1])
        
        # M7 = (a12 - a22)(b21 + b22)
        # U: a12 (2), a22 (3)
        # V: b21 (10+k0), b22 (10+k1)
        # W: C11 (-1), C21 (+1)
        add_term([2, 3], [10+k0, 10+k1], [k0*2+0, k0*2+1], [1, -1], [1, 1], [-1, 1])
        
        # --- Linear Part (Rank 4) ---
        # a3 (col 2 of A) * b3 (row 2 of B)
        # A indices: i in {0,1}, j=2
        # B indices: j=2, k in {k0, k1}
        # C indices: i in {0,1}, k in {k0, k1}
        
        # Terms:
        # a13 b31 -> C11
        # a13 b32 -> C12
        # a23 b31 -> C21
        # a23 b32 -> C22
        
        # a13 is U row 4 (i=0, j=2)
        # a23 is U row 5 (i=1, j=2)
        # b31 is V row 2*10+k0 = 20+k0
        # b32 is V row 2*10+k1 = 20+k1
        
        # C11: W row k0*2+0
        # C12: W row k1*2+0
        # C21: W row k0*2+1
        # C22: W row k1*2+1
        
        add_term([4], [20+k0], [k0*2+0], [1], [1], [1])
        add_term([4], [20+k1], [k1*2+0], [1], [1], [1])
        add_term([5], [20+k0], [k0*2+1], [1], [1], [1])
        add_term([5], [20+k1], [k1*2+1], [1], [1], [1])
        
    return {"U": U, "V": V, "W": W}
