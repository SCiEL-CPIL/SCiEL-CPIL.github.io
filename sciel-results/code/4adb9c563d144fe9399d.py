import numpy as np
import time
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 3
    
    # Rank 26 Decomposition
    # 7 Strassen terms for 2x2 top-left
    # 4 cross terms for A[i,2]*B[2,j]
    # 15 schoolbook terms for the rest
    
    terms = []
    
    # 1. Strassen terms for indices 0,1
    # Local indices 0,1 correspond to global 0,1
    
    # M1: (A00+A11)*(B00+B11) -> C00, C11
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2: (A10+A11)*B00 -> C10
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1}
    ))
    
    # M3: A00*(B10-B11) -> C11
    terms.append((
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(1,1): 1}
    ))
    
    # M4: A11*(B01-B00) -> C00
    terms.append((
        {(1,1): 1},
        {(0,1): 1, (0,0): -1},
        {(0,0): 1}
    ))
    
    # M5: (A00+A01)*B11 -> C01
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,1): 1}
    ))
    
    # M6: (A10-A00)*(B00+B01) -> C10
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1}
    ))
    
    # M7: (A01-A11)*(B10+B11) -> C01
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,1): 1}
    ))
    
    # 2. Cross terms A[i,2] * B[2,j] for i,j in {0,1}
    
    # A02 * B20 -> C00
    terms.append((
        {(0,2): 1},
        {(2,0): 1},
        {(0,0): 1}
    ))
    
    # A02 * B21 -> C01
    terms.append((
        {(0,2): 1},
        {(2,1): 1},
        {(0,1): 1}
    ))
    
    # A12 * B20 -> C10
    terms.append((
        {(1,2): 1},
        {(2,0): 1},
        {(1,0): 1}
    ))
    
    # A12 * B21 -> C11
    terms.append((
        {(1,2): 1},
        {(2,1): 1},
        {(1,1): 1}
    ))
    
    # 3. Schoolbook terms for C02, C12, C20, C21, C22
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    terms.append((
        {(0,0): 1},
        {(0,2): 1},
        {(0,2): 1}
    ))
    terms.append((
        {(0,1): 1},
        {(1,2): 1},
        {(0,2): 1}
    ))
    terms.append((
        {(0,2): 1},
        {(2,2): 1},
        {(0,2): 1}
    ))
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    terms.append((
        {(1,0): 1},
        {(0,2): 1},
        {(1,2): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,2): 1},
        {(1,2): 1}
    ))
    terms.append((
        {(1,2): 1},
        {(2,2): 1},
        {(1,2): 1}
    ))
    
    # C20 = A20*B00 + A21*B10 + A22*B20
    terms.append((
        {(2,0): 1},
        {(0,0): 1},
        {(2,0): 1}
    ))
    terms.append((
        {(2,1): 1},
        {(1,0): 1},
        {(2,0): 1}
    ))
    terms.append((
        {(2,2): 1},
        {(2,0): 1},
        {(2,0): 1}
    ))
    
    # C21 = A20*B01 + A21*B11 + A22*B21
    terms.append((
        {(2,0): 1},
        {(0,1): 1},
        {(2,1): 1}
    ))
    terms.append((
        {(2,1): 1},
        {(1,1): 1},
        {(2,1): 1}
    ))
    terms.append((
        {(2,2): 1},
        {(2,1): 1},
        {(2,1): 1}
    ))
    
    # C22 = A20*B02 + A21*B12 + A22*B22
    terms.append((
        {(2,0): 1},
        {(0,2): 1},
        {(2,2): 1}
    ))
    terms.append((
        {(2,1): 1},
        {(1,2): 1},
        {(2,2): 1}
    ))
    terms.append((
        {(2,2): 1},
        {(2,2): 1},
        {(2,2): 1}
    ))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (i, k), val in w_dict.items():
            W[k * n + i][r] = val
            
    # Verification
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, size=(n, m))
    B_test = np.random.randint(-5, 6, size=(m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_val = sum(U[i * m + j][r] * A_test[i, j] for i in range(n) for j in range(m))
        v_val = sum(V[j * p + k][r] * B_test[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += u_val * v_val * W[k * n + i][r]
    
    if not np.allclose(C_computed, C_expected):
        # Fallback to Schoolbook Rank 27 if something went wrong
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
