import time
import numpy as np

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Strassen 2x2x2 decomposition
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(g)
    # M3 = (a-b)(e+f)
    # M4 = (a+d)(f-h)
    # M5 = (a+c)(h)
    # M6 = (b+d)(g+h)
    # M7 = (a+b)(e)
    # 
    # C11 = M1 - M4 + M6
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3
    #
    # U (A factors):
    # M1: a+d -> [1, 1, 1, 1]
    # M2: c+d -> [0, 0, 1, 1]
    # M3: a-b -> [1, -1, 0, 0]
    # M4: a+d -> [1, 0, 1, 0] ? No.
    # Let's use the standard Strassen coefficients.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)g
    # M3 = (a-b)(e+f)
    # M4 = (a+d)(f-h)
    # M5 = (a+c)h
    # M6 = (b+d)(g+h)
    # M7 = (a+b)e
    #
    # C = [[M1 - M4 + M6, M3 + M4],
    #      [M2 + M5, M1 - M2 + M3]]
    #
    # U rows (i, j) -> coeff for M_r
    # i=0, j=0 (a): M1:1, M3:1, M4:1, M5:1, M7:1
    # i=0, j=1 (b): M3:-1, M6:1, M7:1
    # i=1, j=0 (c): M2:1, M5:1
    # i=1, j=1 (d): M1:1, M2:1, M4:1, M6:1
    #
    # V rows (j, k) -> coeff for M_r
    # j=0, k=0 (e): M1:1, M3:1, M7:1
    # j=0, k=1 (f): M3:1, M4:1
    # j=1, k=0 (g): M2:1, M6:1
    # j=1, k=1 (h): M1:1, M4:-1, M5:1, M6:1
    #
    # W rows (k, i) -> coeff for M_r
    # k=0, i=0 (C00): M1:1, M4:-1, M6:1
    # k=1, i=0 (C01): M3:1, M4:1
    # k=0, i=1 (C10): M2:1, M5:1
    # k=1, i=1 (C11): M1:1, M2:-1, M3:1
    #
    # Note: W index is k*n + i. n=2.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    #
    strassen_U = [
        [1, 0, 1, 1, 1, 0, 1], # a
        [0, 1, -1, 0, 0, 1, 1], # b
        [0, 1, 0, 0, 1, 0, 0], # c
        [1, 1, 0, 1, 0, 1, 0]  # d
    ]
    strassen_V = [
        [1, 0, 1, 1, 0, 0, 1], # e
        [0, 0, 1, 1, 0, 0, 0], # f
        [0, 1, 0, 0, 0, 1, 0], # g
        [1, 0, 0, -1, 1, 1, 0] # h
    ]
    strassen_W = [
        [1, 0, 0, -1, 0, 1, 0], # C00 (k=0, i=0)
        [0, 0, 1, 1, 0, 0, 0], # C01 (k=1, i=0) -> Index 2
        [0, 1, 0, 0, 1, 0, 0], # C10 (k=0, i=1) -> Index 1
        [1, -1, 1, 0, 0, 0, 0] # C11 (k=1, i=1) -> Index 3
    ]
    # Wait, the order of W in the list above corresponds to indices 0, 2, 1, 3?
    # Let's be careful.
    # W index = k*n + i.
    # k=0, i=0 -> 0. C00.
    # k=0, i=1 -> 1. C10.
    # k=1, i=0 -> 2. C01.
    # k=1, i=1 -> 3. C11.
    
    # Let's re-verify the W coefficients.
    # C00 = M1 - M4 + M6. Coeffs: [1, 0, 0, -1, 0, 1, 0]
    # C10 = M2 + M5. Coeffs: [0, 1, 0, 0, 1, 0, 0]
    # C01 = M3 + M4. Coeffs: [0, 0, 1, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3. Coeffs: [1, -1, 1, 0, 0, 0, 0]
    
    strassen_W = [
        [1, 0, 0, -1, 0, 1, 0], # idx 0 (k=0, i=0)
        [0, 1, 0, 0, 1, 0, 0], # idx 1 (k=0, i=1)
        [0, 0, 1, 1, 0, 0, 0], # idx 2 (k=1, i=0)
        [1, -1, 1, 0, 0, 0, 0] # idx 3 (k=1, i=1)
    ]
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    col = 0
    # 4 Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                # U: i in 0,1; j in j0, j0+1
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                # V: j in j0, j0+1; k in k0, k0+1
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                # W: k in k0, k0+1; i in 0,1
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
                
    # 2 Schoolbook blocks for k=4
    # j in 0,1 and j in 2,3
    for j0 in [0, 2]:
        for i in range(n):
            for j in range(j0, j0+2):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
