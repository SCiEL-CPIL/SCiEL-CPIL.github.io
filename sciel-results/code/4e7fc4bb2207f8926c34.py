"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for A (2x2)
    # Indices: i*m + j
    u_vecs = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 1, 1, 0], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 1, 0, 0], # M6: A21 - A11
        [0, -1, 1, 0]  # M7: A12 - A22
    ]
    
    for r in range(5):
        base_col = r * 7
        # B block columns: 2r, 2r+1
        # V indices: j*p + k
        # k1 = 2r, k2 = 2r+1
        # j1 = 0, j2 = 1
        # v_vecs for this block
        # M1: B11 + B22 -> V[0*p+2r]=1, V[1*p+2r+1]=1
        # M2: B11 -> V[0*p+2r]=1
        # M3: B12 - B22 -> V[0*p+2r+1]=1, V[1*p+2r+1]=-1
        # M4: B21 - B11 -> V[1*p+2r]=1, V[0*p+2r]=-1
        # M5: B22 -> V[1*p+2r+1]=1
        # M6: B11 + B12 -> V[0*p+2r]=1, V[0*p+2r+1]=1
        # M7: B21 + B22 -> V[1*p+2r]=1, V[1*p+2r+1]=1
        
        v_sparsity = [
            {0*p + 2*r: 1, 1*p + 2*r + 1: 1},
            {0*p + 2*r: 1},
            {0*p + 2*r + 1: 1, 1*p + 2*r + 1: -1},
            {1*p + 2*r: 1, 0*p + 2*r: -1},
            {1*p + 2*r + 1: 1},
            {0*p + 2*r: 1, 0*p + 2*r + 1: 1},
            {1*p + 2*r: 1, 1*p + 2*r + 1: 1}
        ]
        
        # W indices: k*n + i
        # C block columns: 2r, 2r+1
        # k1 = 2r, k2 = 2r+1
        # i1 = 0, i2 = 1
        # C11 (i=0, k=2r) -> W[2r*2 + 0]
        # C12 (i=0, k=2r+1) -> W[(2r+1)*2 + 0]
        # C21 (i=1, k=2r) -> W[2r*2 + 1]
        # C22 (i=1, k=2r+1) -> W[(2r+1)*2 + 1]
        
        w_sparsity = [
            {2*r*2 + 0: 1, (2*r+1)*2 + 1: 1}, # M1: C11 + C22
            {2*r*2 + 1: 1, (2*r+1)*2 + 1: -1}, # M2: C21 - C22
            {(2*r+1)*2 + 0: 1, (2*r+1)*2 + 1: 1}, # M3: C12 + C22
            {2*r*2 + 0: 1, 2*r*2 + 1: 1}, # M4: C11 + C21
            {2*r*2 + 0: -1, (2*r+1)*2 + 0: 1}, # M5: -C11 + C12
            {(2*r+1)*2 + 1: 1}, # M6: C22
            {2*r*2 + 0: 1} # M7: C11
        ]
        
        for s in range(7):
            col = base_col + s
            # Fill U
            for i in range(4):
                U[i][col] = u_vecs[s][i]
            # Fill V
            for idx, val in v_sparsity[s].items():
                V[idx][col] = val
            # Fill W
            for idx, val in w_sparsity[s].items():
                W[idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
