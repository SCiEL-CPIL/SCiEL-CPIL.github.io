"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition for 4x4 matrix multiplication using Strassen's algorithm.
    # Strassen rank for 2x2 is 7.
    # Recursive application gives rank 7^2 = 49 for 4x4.
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen decomposition for 2x2 matrices
    # Indices 0..3 correspond to:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # u_l, v_l, w_l for l=0..6
    u_inner = [
        [1, 0, 0, 1], # 1: x11 + x22
        [0, 0, 1, 1], # 2: x21 + x22
        [1, 0, 0, 0], # 3: x11
        [0, 0, 0, 1], # 4: x22
        [1, 1, 0, 0], # 5: x11 + x12
        [0, 0, 1, -1],# 6: x21 - x11
        [0, 1, 0, -1] # 7: x12 - x22
    ]
    
    v_inner = [
        [1, 0, 0, 1], # 1: y11 + y22
        [1, 0, 0, 0], # 2: y11
        [0, 1, 0, -1],# 3: y12 - y22
        [0, 0, 1, -1],# 4: y21 - y11
        [0, 0, 0, 1], # 5: y22
        [1, 1, 0, 0], # 6: y11 + y12
        [0, 0, 1, 1]  # 7: y21 + y22
    ]
    
    w_inner = [
        [1, 0, 0, 1], # 1: z11 + z22
        [0, 0, 1, -1],# 2: z21 - z22
        [0, 1, 0, 1], # 3: z12 + z22
        [1, 0, 1, 0], # 4: z11 + z21
        [-1, 1, 0, 0],# 5: -z11 + z12
        [0, 0, 0, 1], # 6: z22
        [1, 0, 0, 0]  # 7: z11
    ]
    
    # Outer Strassen coefficients for 4x4 block multiplication
    # Blocks: 0: 11, 1: 12, 2: 21, 3: 22
    u_outer = [
        [1, 0, 0, 1], # 1: A11 + A22
        [0, 0, 1, 1], # 2: A21 + A22
        [1, 0, 0, 0], # 3: A11
        [0, 0, 0, 1], # 4: A22
        [1, 1, 0, 0], # 5: A11 + A12
        [0, 0, 1, -1],# 6: A21 - A11
        [0, 1, 0, -1] # 7: A12 - A22
    ]
    
    v_outer = [
        [1, 0, 0, 1], # 1: B11 + B22
        [1, 0, 0, 0], # 2: B11
        [0, 1, 0, -1],# 3: B12 - B22
        [0, 0, 1, -1],# 4: B21 - B11
        [0, 0, 0, 1], # 5: B22
        [1, 1, 0, 0], # 6: B11 + B12
        [0, 0, 1, 1]  # 7: B21 + B22
    ]
    
    w_outer = [
        [1, 0, 0, 1], # 1: C11 + C22
        [0, 0, 1, -1],# 2: C21 - C22
        [0, 1, 0, 1], # 3: C12 + C22
        [1, 0, 1, 0], # 4: C11 + C21
        [-1, 1, 0, 0],# 5: -C11 + C12
        [0, 0, 0, 1], # 6: C22
        [1, 0, 0, 0]  # 7: C11
    ]
    
    # Helper to get global index for A or B
    # Block b (0-3), local row r (0-1), local col c (0-1)
    def get_ab_index(b, r, c):
        if b == 0: # 11
            return r * 4 + c
        elif b == 1: # 12
            return r * 4 + (2 + c)
        elif b == 2: # 21
            return (2 + r) * 4 + c
        else: # 22
            return (2 + r) * 4 + (2 + c)
            
    # Helper to get global index for C
    # Block b (0-3), local row r (0-1), local col c (0-1)
    # C index convention: W[k*n + i] where i is row, k is col.
    # So index = c * 4 + r
    def get_c_index(b, r, c):
        if b == 0: # 11
            return c * 4 + r
        elif b == 1: # 12
            return (2 + c) * 4 + r
        elif b == 2: # 21
            return c * 4 + (2 + r)
        else: # 22
            return (2 + c) * 4 + (2 + r)

    col = 0
    for l in range(7):
        for lp in range(7):
            # U contribution
            for b in range(4):
                alpha = u_outer[l][b]
                if alpha == 0: continue
                for r in range(2):
                    for c in range(2):
                        coeff = alpha * u_inner[lp][r * 2 + c]
                        if coeff != 0:
                            idx = get_ab_index(b, r, c)
                            U[idx][col] += coeff
                            
            # V contribution
            for b in range(4):
                beta = v_outer[l][b]
                if beta == 0: continue
                for r in range(2):
                    for c in range(2):
                        coeff = beta * v_inner[lp][r * 2 + c]
                        if coeff != 0:
                            idx = get_ab_index(b, r, c)
                            V[idx][col] += coeff
                            
            # W contribution
            for b in range(4):
                gamma = w_outer[l][b]
                if gamma == 0: continue
                for r in range(2):
                    for c in range(2):
                        coeff = gamma * w_inner[lp][r * 2 + c]
                        if coeff != 0:
                            idx = get_c_index(b, r, c)
                            W[idx][col] += coeff
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
