import time

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (a,b,c,d)
    # P1: a+d
    # P2: c+d
    # P3: a
    # P4: d
    # P5: a+b
    # P6: c-a
    # P7: b-d
    A_coeffs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [1, 0, -1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    # Strassen coefficients for B block (e,f,g,h)
    # P1: e+h
    # P2: e
    # P3: f-h
    # P4: g-e
    # P5: h
    # P6: e+f
    # P7: g+h
    B_coeffs = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # Output coefficients for C block (c00, c01, c10, c11)
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    C_coeffs = [
        [1, 0, 0, 0], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    for l in range(num_blocks):
        for s in range(7):
            col_idx = l * 7 + s
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    U[u_idx][col_idx] = A_coeffs[s][i * m + j]
                    
            # Fill V
            # B local indices: e(0,0), f(0,1), g(1,0), h(1,1)
            # Global B index: j * p + k
            # e: j=0, k=2l -> 2l
            # f: j=0, k=2l+1 -> 2l+1
            # g: j=1, k=2l -> p + 2l
            # h: j=1, k=2l+1 -> p + 2l + 1
            
            b_indices = [2*l, 2*l+1, p+2*l, p+2*l+1]
            for local_j in range(2):
                for local_k in range(2):
                    v_idx = b_indices[local_j * 2 + local_k]
                    V[v_idx][col_idx] = B_coeffs[s][local_j * 2 + local_k]
                    
            # Fill W
            # C local indices: c00(0,0), c01(0,1), c10(1,0), c11(1,1)
            # Global C index: k * n + i
            # c00: i=0, k=2l -> 2l * n + 0
            # c01: i=0, k=2l+1 -> (2l+1) * n + 0
            # c10: i=1, k=2l -> 2l * n + 1
            # c11: i=1, k=2l+1 -> (2l+1) * n + 1
            
            c_indices = [
                2*l * n + 0,
                (2*l+1) * n + 0,
                2*l * n + 1,
                (2*l+1) * n + 1
            ]
            
            for local_k in range(2):
                for local_i in range(n):
                    w_idx = c_indices[local_k * n + local_i]
                    W[w_idx][col_idx] = C_coeffs[s][local_k * n + local_i]
                    
    return {"U": U, "V": V, "W": W}
