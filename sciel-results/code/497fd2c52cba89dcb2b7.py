import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # 7 Strassen blocks of 2x2, 1 standard block of 2x1
    num_strassen = 7
    num_standard = 1
    
    rank = num_strassen * 7 + num_standard * 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Strassen blocks
    for b in range(num_strassen):
        # Local indices i, j, k in {0, 1}
        # Global j = 2*b + j_local
        # Global k = 2*b + k_local
        
        # Strassen factors
        # P1: (a+d)(e+h)
        u1 = np.array([[1, 0], [1, 1]])
        v1 = np.array([[1, 0], [0, 1]])
        w1 = np.array([[1, 0], [0, 1]]) # w[k, i]
        
        # P2: (c+d)e
        u2 = np.array([[0, 0], [1, 1]])
        v2 = np.array([[1, 0], [0, 0]])
        w2 = np.array([[0, 0], [1, 0]]) # C10 += P2 -> w[0, 1]? No.
        # C10 is i=1, k=0. w[k, i] = w[0, 1].
        
        # P3: a(f-h)
        u3 = np.array([[1, 0], [0, 0]])
        v3 = np.array([[0, 1], [0, -1]])
        w3 = np.array([[0, 1], [0, 0]]) # C01 += P3 -> i=0, k=1. w[1, 0].
        
        # P4: d(g-f)
        u4 = np.array([[0, 0], [0, 1]])
        v4 = np.array([[0, -1], [1, 0]])
        w4 = np.array([[1, 0], [0, 1]]) # C00 += P4 (w[0,0]), C10 += P4 (w[0,1]).
        # Wait, C10 is i=1, k=0. w[0, 1].
        # C00 is i=0, k=0. w[0, 0].
        
        # P5: (a+b)h
        u5 = np.array([[1, 1], [0, 0]])
        v5 = np.array([[0, 0], [0, 1]])
        w5 = np.array([[0, -1], [0, 1]]) # C00 -= P5 (w[0,0]=-1), C01 += P5 (w[1,0]=1).
        
        # P6: (c-a)(e+f)
        u6 = np.array([[-1, 0], [1, 0]])
        v6 = np.array([[1, 1], [0, 0]])
        w6 = np.array([[0, 0], [0, 1]]) # C11 += P6 (w[1,1]=1).
        
        # P7: (b-d)(g+h)
        u7 = np.array([[0, 1], [0, -1]])
        v7 = np.array([[0, 0], [1, 1]])
        w7 = np.array([[1, 0], [0, 0]]) # C00 += P7 (w[0,0]=1).
        
        factors = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u, v, w in factors:
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j, col] = u[i, j]
            
            # Fill V
            for j in range(m):
                for k in range(2): # local k
                    V[j * p + (2 * b + k), col] = v[j, k]
            
            # Fill W
            for k in range(2): # local k
                for i in range(n):
                    W[(2 * b + k) * n + i, col] = w[k, i]
            
            col += 1
    
    # Standard block (2x1) at j=14, k=14
    # 4 terms: ax, by, cx, dy
    std_factors = [
        # ax -> C0
        (np.array([[1, 0], [0, 0]]), np.array([[1], [0]]), np.array([[1], [0]])),
        # by -> C0
        (np.array([[0, 1], [0, 0]]), np.array([[0], [1]]), np.array([[1], [0]])),
        # cx -> C1
        (np.array([[0, 0], [1, 0]]), np.array([[1], [0]]), np.array([[0], [1]])),
        # dy -> C1
        (np.array([[0, 0], [0, 1]]), np.array([[0], [1]]), np.array([[0], [1]]))
    ]
    
    for u, v, w in std_factors:
        # Fill U
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = u[i, j]
        
        # Fill V
        for j in range(m):
            # k is 0 locally, global k = 14
            V[j * p + 14, col] = v[j, 0]
        
        # Fill W
        # k is 0 locally, global k = 14
        for i in range(n):
            W[14 * n + i, col] = w[0, i]
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
