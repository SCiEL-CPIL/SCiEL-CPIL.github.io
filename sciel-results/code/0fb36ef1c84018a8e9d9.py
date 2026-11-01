import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    # Ranks
    # 4 blocks of 2x2 (28 terms) + 1 block of 2x1 (4 terms) = 32
    rank = 32
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen for 2x2 blocks
    for b in range(4):
        # B indices
        idx_e = 0 * p + 2 * b
        idx_f = 0 * p + 2 * b + 1
        idx_g = 1 * p + 2 * b
        idx_h = 1 * p + 2 * b + 1
        
        # C indices
        idx_c00 = (2 * b) * n + 0
        idx_c01 = (2 * b + 1) * n + 0
        idx_c10 = (2 * b) * n + 1
        idx_c11 = (2 * b + 1) * n + 1
        
        # m1 = (a+d)(e+h)
        U[0, col] = 1; U[3, col] = 1
        V[idx_e, col] = 1; V[idx_h, col] = 1
        W[idx_c00, col] = 1; W[idx_c11, col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[2, col] = 1; U[3, col] = 1
        V[idx_e, col] = 1
        W[idx_c10, col] = 1; W[idx_c11, col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[0, col] = 1
        V[idx_f, col] = 1; V[idx_h, col] = -1
        W[idx_c01, col] = 1; W[idx_c11, col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[3, col] = 1
        V[idx_g, col] = 1; V[idx_e, col] = -1
        W[idx_c00, col] = 1; W[idx_c10, col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[idx_h, col] = 1
        W[idx_c00, col] = -1; W[idx_c01, col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[2, col] = 1; U[0, col] = -1
        V[idx_e, col] = 1; V[idx_f, col] = 1
        W[idx_c11, col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[1, col] = 1; U[3, col] = -1
        V[idx_g, col] = 1; V[idx_h, col] = 1
        W[idx_c00, col] = 1
        col += 1
        
    # Standard for 2x1 block (col 8)
    idx_e = 0 * p + 8
    idx_g = 1 * p + 8
    
    idx_c0 = 8 * n + 0
    idx_c1 = 8 * n + 1
    
    # t1 = a * e -> C0
    U[0, col] = 1
    V[idx_e, col] = 1
    W[idx_c0, col] = 1
    col += 1
    
    # t2 = b * g -> C0
    U[1, col] = 1
    V[idx_g, col] = 1
    W[idx_c0, col] = 1
    col += 1
    
    # t3 = c * e -> C1
    U[2, col] = 1
    V[idx_e, col] = 1
    W[idx_c1, col] = 1
    col += 1
    
    # t4 = d * g -> C1
    U[3, col] = 1
    V[idx_g, col] = 1
    W[idx_c1, col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
