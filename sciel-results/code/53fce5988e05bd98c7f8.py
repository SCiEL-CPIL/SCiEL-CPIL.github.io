import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[q][i][j]
    # beta[q][j][k]
    # gamma[q][a][b]
    
    # q=0: M0 = (A00+A11)(B00+B11)
    # q=1: M1 = (A10+A11)B00
    # q=2: M2 = A00(B01-B11)
    # q=3: M3 = A11(B10-B00)
    # q=4: M4 = (A00+A01)B11
    # q=5: M5 = (A10-A00)(B00+B01)
    # q=6: M6 = (A01-A11)(B10+B11)
    
    alphas = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    betas = [
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    gammas = [
        { (0,0): 1, (1,1): 1 }, # M0 in C00, C11
        { (1,0): 1, (1,1): -1 }, # M1 in C10, C11
        { (0,1): 1, (1,1): 1 },  # M2 in C01, C11
        { (0,0): 1, (1,0): 1 },  # M3 in C00, C10
        { (0,0): -1, (0,1): 1 }, # M4 in C00, C01
        { (1,1): 1 },            # M5 in C11
        { (0,0): 1 }             # M6 in C00
    ]
    
    col = 0
    for q in range(7):
        a_dict = alphas[q]
        b_dict = betas[q]
        g_dict = gammas[q]
        
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # Update U
                    for (i, j), val in a_dict.items():
                        row_U = (i * 3 + u) * 6 + (j * 3 + v)
                        U[row_U][col] += val
                        
                    # Update V
                    for (j, k), val in b_dict.items():
                        row_V = (j * 3 + v) * 6 + (k * 3 + w)
                        V[row_V][col] += val
                        
                    # Update W
                    for (a, b), val in g_dict.items():
                        # W index: k_global * n + i_global
                        # i_global = a * 3 + u
                        # k_global = b * 3 + w
                        row_W = (b * 3 + w) * 6 + (a * 3 + u)
                        W[row_W][col] += val
                        
                    col += 1
                    if time.time() > deadline:
                        # Return best so far? 
                        # Since it's deterministic and fast, this shouldn't happen.
                        # But if it does, we might have incomplete matrices.
                        # Better to just let it finish as it's very fast.
                        pass

    return {"U": U, "V": V, "W": W}
