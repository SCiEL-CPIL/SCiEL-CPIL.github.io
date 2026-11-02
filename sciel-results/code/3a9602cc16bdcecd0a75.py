"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 4 blocks
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_part[i, k, s], V_part[k, l, s], W_part[l, i, s]
    # s in 0..6
    
    # M1: (a+d)(e+h)
    # U: a(0,0)=1, d(1,1)=1
    # V: e(0,0)=1, h(1,1)=1
    # M2: (c+d)e
    # U: c(1,0)=1, d(1,1)=1
    # V: e(0,0)=1
    # M3: a(f-h)
    # U: a(0,0)=1
    # V: f(0,1)=1, h(1,1)=-1
    # M4: d(g-e)
    # U: d(1,1)=1
    # V: g(1,0)=1, e(0,0)=-1
    # M5: (a+b)h
    # U: a(0,0)=1, b(0,1)=1
    # V: h(1,1)=1
    # M6: (c-a)(e+f)
    # U: c(1,0)=1, a(0,0)=-1
    # V: e(0,0)=1, f(0,1)=1
    # M7: (b-d)(g+h)
    # U: b(0,1)=1, d(1,1)=-1
    # V: g(1,0)=1, h(1,1)=1
    
    # W coefficients for C_il = sum_s W[l, i, s] M_s
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_coeffs = [
        [1, 0, 1, 0, 0, 0, 0], # i=0, k=0 (a)
        [0, 0, 0, 0, 1, 0, 1], # i=0, k=1 (b)
        [0, 1, 0, 0, 0, 1, 0], # i=1, k=0 (c)
        [1, 1, 0, 1, 0, 0, -1] # i=1, k=1 (d)
    ]
    
    V_coeffs = [
        [1, 1, 0, -1, 0, 1, 0], # k=0, l=0 (e)
        [0, 0, 1, 0, 0, 1, 0],  # k=0, l=1 (f)
        [0, 0, 0, 1, 0, 0, 1],  # k=1, l=0 (g)
        [1, 0, -1, 0, 1, 0, 1]  # k=1, l=1 (h)
    ]
    
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # l=0, i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],  # l=1, i=0 (C01)
        [0, 1, 0, 1, 0, 0, 0],  # l=0, i=1 (C10)
        [1, -1, 1, 0, 0, 1, 0]  # l=1, i=1 (C11)
    ]
    
    # Construct global matrices
    # r = 7*j + s
    for j in range(4):
        for s in range(7):
            r = 7 * j + s
            
            # Fill U
            for i in range(2):
                for k in range(2):
                    U[i * m + k][r] = U_coeffs[i * 2 + k][s]
                    
            # Fill V
            # Global column index L = 2*j + l
            for k in range(2):
                for l in range(2):
                    L = 2 * j + l
                    V[k * p + L][r] = V_coeffs[k * 2 + l][s]
                    
            # Fill W
            # Global row index of W is L * n + i
            for l in range(2):
                for i in range(2):
                    L = 2 * j + l
                    W[L * n + i][r] = W_coeffs[l * 2 + i][s]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
