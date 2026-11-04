import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 36
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U factors (coeff of A[i,j])
    # i,j in {0,1}
    U_factors = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    # V factors (coeff of B[j,k])
    # j,k in {0,1}
    V_factors = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    # W factors (coeff of C[i,k])
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_factors = [
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }, # M1
        { (0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1 }, # M2
        { (0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1 },  # M3
        { (0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0 },  # M4
        { (0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0 }, # M5
        { (0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1 },  # M6
        { (0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0 }   # M7
    ]
    
    # Fill 5 blocks of Strassen
    for block in range(5):
        k_base = block * 2
        for s in range(7):
            r = block * 7 + s
            
            # Fill U
            for (i, j), val in U_factors[s].items():
                U[i * m + j, r] += val
            
            # Fill V
            for (j, k), val in V_factors[s].items():
                V[j * p + k_base + k, r] += val
            
            # Fill W
            for (i, k), val in W_factors[s].items():
                W[(k_base + k) * n + i, r] += val
    
    # Fill remainder for j=2
    r_rem = 35
    for i in range(n):
        U[i * m + 2, r_rem] = 1.0
    for k in range(p):
        V[2 * p + k, r_rem] = 1.0
    for k in range(p):
        for i in range(n):
            W[k * n + i, r_rem] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
