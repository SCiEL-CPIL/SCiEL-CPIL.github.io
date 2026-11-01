import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, safe margin

    n, m, p = 2, 2, 14
    num_blocks = p // 2  # 7 blocks of size 2
    rank = num_blocks * 7  # 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms: (a, b, c, d, e, f, g, h, c00, c01, c10, c11)
    # A = [[a,b],[c,d]], B_block = [[e,f],[g,h]], C_block = [[c00,c01],[c10,c11]]
    strassen_terms = [
        # M1 = (a+d)(e+h): C00+=1, C11+=1
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (c+d)e: C10+=1, C11-=1
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        # M3 = a(f-h): C01+=1, C11+=1
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        # M4 = d(g-e): C00+=1, C10+=1
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        # M5 = (a+b)h: C00-=1, C01+=1
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (c-a)(e+f): C11+=1
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        # M7 = (b-d)(g+h): C00+=1
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    ]

    for block in range(num_blocks):
        for t_idx, term in enumerate(strassen_terms):
            col = block * 7 + t_idx
            a_c, b_c, c_c, d_c, e_c, f_c, g_c, h_c, c00_c, c01_c, c10_c, c11_c = term

            # U: A part, index i*m+j
            U[0][col] = a_c  # A[0,0]
            U[1][col] = b_c  # A[0,1]
            U[2][col] = c_c  # A[1,0]
            U[3][col] = d_c  # A[1,1]

            # V: B part, index j*p+k
            V[0 * p + 2 * block][col] = e_c       # B[0, 2*block]
            V[0 * p + 2 * block + 1][col] = f_c   # B[0, 2*block+1]
            V[1 * p + 2 * block][col] = g_c       # B[1, 2*block]
            V[1 * p + 2 * block + 1][col] = h_c   # B[1, 2*block+1]

            # W: C part, index k*n+i (trace convention W[k*n+i])
            W[(2 * block) * n + 0][col] = c00_c       # C[0, 2*block]
            W[(2 * block) * n + 1][col] = c10_c       # C[1, 2*block]
            W[(2 * block + 1) * n + 0][col] = c01_c   # C[0, 2*block+1]
            W[(2 * block + 1) * n + 1][col] = c11_c   # C[1, 2*block+1]

    return {"U": U, "V": V, "W": W}
