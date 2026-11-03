import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try Laderman + Schoolbook
    R = 32
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    laderman_terms = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        ({(0,1): 1, (0,0): -1}, {(0,0): 1}, {(0,0): -1}),
        ({(1,1): 1}, {(0,0): 1, (1,1): 1}, {(1,1): 1}),
        ({(1,1): 1, (0,0): 1}, {(1,0): 1}, {(1,0): 1}),
        ({(0,0): 1, (1,1): 1}, {(0,1): 1}, {(0,1): 1}),
        ({(0,1): 1, (1,1): 1}, {(1,0): 1}, {(0,0): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1}),
        ({(1,0): 1, (0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,0): 1}, {(0,0): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,0): 1}, {(1,1): 1}),
        ({(0,1): 1}, {(1,0): 1, (1,1): 1}, {(0,1): 1}),
        ({(1,0): 1}, {(0,0): 1, (0,1): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (0,1): 1}, {(0,0): 1}),
        ({(1,1): 1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        ({(0,1): 1}, {(0,0): 1, (0,1): 1}, {(0,1): 1}),
        ({(1,0): 1}, {(1,1): 1, (0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,1): 1, (0,0): 1}, {(0,0): 1}),
        ({(1,1): 1}, {(1,1): 1, (0,0): 1}, {(1,1): 1}),
        ({(0,1): 1}, {(1,1): 1, (1,0): 1}, {(0,1): 1}),
        ({(1,0): 1}, {(0,1): 1, (1,1): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r in range(23):
        coeff_A, coeff_B, coeff_C = laderman_terms[r]
        for (i, j), val in coeff_A.items():
            U[i*m + j, r] += val
        for (j, k), val in coeff_B.items():
            V[j*p + k, r] += val
        for (i, k), val in coeff_C.items():
            W[k*n + i, r] += val
            
    for i in range(3):
        for j in range(3):
            r = 23 + i*3 + j
            U[i*m + j, r] = 1
            V[j*p + 3, r] = 1
            W[3*n + i, r] = 1
            
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if not _verify_decomp(U_list, V_list, W_list, n, m, p):
        return schoolbook()
        
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomp(U, V, W, n, m, p):
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                s = 0
                for r in range(R):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1) > 1e-6:
                    return False
    for a in range(n*m):
        i1 = a // m
        j1 = a % m
        for b in range(m*p):
            j2 = b // p
            k2 = b % p
            if j1 != j2:
                for c in range(p*n):
                    s = 0
                    for r in range(R):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-6:
                        return False
            else:
                for c in range(p*n):
                    k1 = c // n
                    i2 = c % n
                    if i1 == i2 and k1 == k2:
                        continue
                    else:
                        s = 0
                        for r in range(R):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if abs(s) > 1e-6:
                            return False
    return True
