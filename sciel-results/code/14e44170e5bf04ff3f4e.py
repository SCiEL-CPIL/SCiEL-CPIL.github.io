import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms for 2x2 multiplication
    # M = [[a, b], [c, d]], N = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # We define the 6 instances of 2x2 multiplication
    # Each instance is defined by:
    # a_idx: (row, col) indices in A (flattened i*m+j)
    # b_idx: (row, col) indices in B (flattened j*p+k)
    # c_idx: (row, col) indices in C (flattened k*n+i)
    
    # A is 2x4. Indices:
    # A1 (cols 0,1): 
    #   a: (0,0)=0, b: (0,1)=1
    #   c: (1,0)=2, d: (1,1)=3
    # A2 (cols 2,3):
    #   a: (0,2)=2, b: (0,3)=3
    #   c: (1,2)=4, d: (1,3)=5
    
    # B is 4x6. Indices:
    # B11 (rows 0,1; cols 0,1):
    #   e: (0,0)=0, f: (0,1)=1
    #   g: (1,0)=6, h: (1,1)=7
    # B21 (rows 2,3; cols 0,1):
    #   e: (2,0)=12, f: (2,1)=13
    #   g: (3,0)=18, h: (3,1)=19
    # B12 (rows 0,1; cols 2,3):
    #   e: (0,2)=2, f: (0,3)=3
    #   g: (1,2)=8, h: (1,3)=9
    # B22 (rows 2,3; cols 2,3):
    #   e: (2,2)=14, f: (2,3)=15
    #   g: (3,2)=20, h: (3,3)=21
    # B13 (rows 0,1; cols 4,5):
    #   e: (0,4)=4, f: (0,5)=5
    #   g: (1,4)=10, h: (1,5)=11
    # B23 (rows 2,3; cols 4,5):
    #   e: (2,4)=16, f: (2,5)=17
    #   g: (3,4)=22, h: (3,5)=23
    
    # C is 6x2 (p x n). Indices k*n+i.
    # C cols 0,1 (k=0,1):
    #   P11: (0,0)=0, P12: (0,1)=2
    #   P21: (1,0)=1, P22: (1,1)=3
    # C cols 2,3 (k=2,3):
    #   P11: (2,0)=4, P12: (2,1)=6
    #   P21: (3,0)=5, P22: (3,1)=7
    # C cols 4,5 (k=4,5):
    #   P11: (4,0)=8, P12: (4,1)=10
    #   P21: (5,0)=9, P22: (5,1)=11

    instances = [
        # (A_indices, B_indices, C_indices)
        # A: [a, b, c, d]
        # B: [e, f, g, h]
        # C: [P11, P12, P21, P22]
        
        # Instance 1: A1 * B11 -> C[:, 0:2]
        ([0, 1, 2, 3], [0, 1, 6, 7], [0, 2, 1, 3]),
        # Instance 2: A2 * B21 -> C[:, 0:2]
        ([2, 3, 4, 5], [12, 13, 18, 19], [0, 2, 1, 3]),
        
        # Instance 3: A1 * B12 -> C[:, 2:4]
        ([0, 1, 2, 3], [2, 3, 8, 9], [4, 6, 5, 7]),
        # Instance 4: A2 * B22 -> C[:, 2:4]
        ([2, 3, 4, 5], [14, 15, 20, 21], [4, 6, 5, 7]),
        
        # Instance 5: A1 * B13 -> C[:, 4:6]
        ([0, 1, 2, 3], [4, 5, 10, 11], [8, 10, 9, 11]),
        # Instance 6: A2 * B23 -> C[:, 4:6]
        ([2, 3, 4, 5], [16, 17, 22, 23], [8, 10, 9, 11]),
    ]
    
    col = 0
    for A_idx, B_idx, C_idx in instances:
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        P11, P12, P21, P22 = C_idx
        
        # m1 = (a+d)(e+h)
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[P11][col] += 1; W[P22][col] += 1
        col += 1
        
        # m2 = (c+d)e
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[P21][col] += 1; W[P22][col] -= 1
        col += 1
        
        # m3 = a(f-h)
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[P12][col] += 1; W[P22][col] += 1
        col += 1
        
        # m4 = d(g-c)
        U[d][col] += 1
        V[g][col] += 1; V[c][col] -= 1
        W[P11][col] += 1; W[P21][col] += 1
        col += 1
        
        # m5 = (a+b)h
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[P11][col] -= 1; W[P12][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[P22][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[P11][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
