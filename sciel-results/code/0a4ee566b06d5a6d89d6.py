import time

def construct():
    """
    Rank-39 exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    
    Strategy:
    - Use Strassen's algorithm (rank 7) for 5 pairs of columns (columns 0-9)
    - Use schoolbook multiplication (rank 4) for the last column (column 10)
    - Total rank: 5*7 + 4 = 39
    
    The decomposition is exact with integer entries.
    """
    n, m, p = 2, 2, 11
    
    rank = 39
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 39)
    V = [[0] * rank for _ in range(m * p)]  # shape (22, 39)
    W = [[0] * rank for _ in range(p * n)]  # shape (22, 39)
    
    r = 0  # current rank index
    
    # A (2x2) indices: 0=a, 1=b, 2=c, 3=d
    # B (2x11) indices: row 0: 0..10, row 1: 11..21
    # C (2x11) -> W[k*n+i, r] where k is column index, i is row index
    
    # Process 5 pairs of columns using Strassen
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # B indices for this 2x2 block
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices (C indices)
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
    
    # Last column (k=10) using schoolbook
    k2 = 10
    v_e2 = k2       # B[0, 10]
    v_g2 = 11 + k2  # B[1, 10]
    w_c02 = 2 * k2     # C[0, 10]
    w_c12 = 2 * k2 + 1 # C[1, 10]
    
    # Term: a * B[0, 10]
    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    # Term: b * B[1, 10]
    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1
    
    # Term: c * B[0, 10]
    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    # Term: d * B[1, 10]
    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
