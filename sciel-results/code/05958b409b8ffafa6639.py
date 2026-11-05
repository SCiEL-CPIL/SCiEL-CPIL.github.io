"""Matrix multiplication 3x4x5 with Strassen-based block decomposition."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Rank-56 exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses Strassen's algorithm (rank 7) for 2x2x2 sub-blocks and schoolbook for the rest.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    def add_term(u, v, w):
        terms.append((dict(u), dict(v), dict(w)))
    
    def schoolbook_term(i, j, k, coeff=1):
        u = {i * m + j: coeff}
        v = {j * p + k: coeff}
        w = {k * n + i: coeff}
        add_term(u, v, w)
    
    def strassen_2x2x2(i0, j0, k0):
        """Strassen decomposition for A[i0:i0+2,j0:j0+2] * B[j0:j0+2,k0:k0+2] -> C[i0:i0+2,k0:k0+2]"""
        # p1 = (a+d)(e+h), contributes to M[0,0] and M[1,1]
        u = {i0*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v = {j0*p + k0: 1, (j0+1)*p + (k0+1): 1}
        w = {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # p2 = (c+d)e, contributes to M[1,0]
        u = {(i0+1)*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v = {j0*p + k0: 1}
        w = {k0*n + (i0+1): 1}
        add_term(u, v, w)
        
        # p3 = a(f-h), contributes to M[0,1] and M[1,1]
        u = {i0*m + j0: 1}
        v = {j0*p + (k0+1): 1, (j0+1)*p + (k0+1): -1}
        w = {(k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # p4 = d(g-e), contributes to M[0,0] and M[1,0]
        u = {(i0+1)*m + (j0+1): 1}
        v = {(j0+1)*p + k0: 1, j0*p + k0: -1}
        w = {k0*n + i0: 1, k0*n + (i0+1): 1}
        add_term(u, v, w)
        
        # p5 = (a+b)h, contributes to M[0,0] (negative) and M[0,1] (positive)
        u = {i0*m + j0: 1, i0*m + (j0+1): 1}
        v = {(j0+1)*p + (k0+1): 1}
        w = {k0*n + i0: -1, (k0+1)*n + i0: 1}
        add_term(u, v, w)
        
        # p6 = (c-a)(e+f), contributes to M[1,1]
        u = {(i0+1)*m + j0: 1, i0*m + j0: -1}
        v = {j0*p + k0: 1, j0*p + (k0+1): 1}
        w = {(k0+1)*n + (i0+1): 1}
        add_term(u, v, w)
        
        # p7 = (b-d)(g+h), contributes to M[0,0]
        u = {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): -1}
        v = {(j0+1)*p + k0: 1, (j0+1)*p + (k0+1): 1}
        w = {k0*n + i0: 1}
        add_term(u, v, w)
    
    # Build decomposition by blocks
    # A rows: [0], [1:3]; A cols: [0:2], [2:4]
    # B rows: [0:2], [2:4]; B cols: [0:2], [2:4], [4:5]
    
    # C[0:1, 0:2] = A[0:1,0:2]*B[0:2,0:2] + A[0:1,2:4]*B[2:4,0:2]
    for j in range(2):
        for k in range(2):
            schoolbook_term(0, j, k)
    for j in range(2, 4):
        for k in range(2):
            schoolbook_term(0, j, k)
    
    # C[0:1, 2:4] = A[0:1,0:2]*B[0:2,2:4] + A[0:1,2:4]*B[2:4,2:4]
    for j in range(2):
        for k in range(2, 4):
            schoolbook_term(0, j, k)
    for j in range(2, 4):
        for k in range(2, 4):
            schoolbook_term(0, j, k)
    
    # C[0:1, 4:5] = A[0:1,0:2]*B[0:2,4:5] + A[0:1,2:4]*B[2:4,4:5]
    for j in range(2):
        for k in range(4, 5):
            schoolbook_term(0, j, k)
    for j in range(2, 4):
        for k in range(4, 5):
            schoolbook_term(0, j, k)
    
    # C[1:3, 0:2] = A[1:3,0:2]*B[0:2,0:2] + A[1:3,2:4]*B[2:4,0:2]
    strassen_2x2x2(1, 0, 0)
    strassen_2x2x2(1, 2, 0)
    
    # C[1:3, 2:4] = A[1:3,0:2]*B[0:2,2:4] + A[1:3,2:4]*B[2:4,2:4]
    strassen_2x2x2(1, 0, 2)
    strassen_2x2x2(1, 2, 2)
    
    # C[1:3, 4:5] = A[1:3,0:2]*B[0:2,4:5] + A[1:3,2:4]*B[2:4,4:5]
    for i in range(1, 3):
        for j in range(2):
            for k in range(4, 5):
                schoolbook_term(i, j, k)
    for i in range(1, 3):
        for j in range(2, 4):
            for k in range(4, 5):
                schoolbook_term(i, j, k)
    
    rank = len(terms)
    
    # Convert to matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][col] = coeff
        for idx, coeff in v.items():
            V[idx][col] = coeff
        for idx, coeff in w.items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
