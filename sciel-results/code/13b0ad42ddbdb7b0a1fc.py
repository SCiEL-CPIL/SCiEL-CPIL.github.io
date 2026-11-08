"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition using Kronecker product of two Strassen (rank-7) decompositions
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen decomposition for 2x2 matrix multiplication
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for 2x2:
    # A: 0:A00, 1:A01, 2:A10, 3:A11
    # B: 0:B00, 1:B01, 2:B10, 3:B11
    # C: 0:C00 (k=0,i=0), 1:C10 (k=0,i=1), 2:C01 (k=1,i=0), 3:C11 (k=1,i=1)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 0,  1, 0], # A01
        [0, 1, 0, 0, 1,  0, 1], # A10
        [1, 1, 0, 1, 0,  0,-1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1,  0, 0, 0, 1], # B01
        [0, 0, 0,  1, 0, 1, 0], # B10
        [1, 0,-1,  0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, -1, -1, 0, 1], # C00 (k=0,i=0)
        [0, 1, 0,  1,  0, 0, 0], # C10 (k=0,i=1)
        [0, 0, 1,  0,  1, 0, 0], # C01 (k=1,i=0)
        [1,-1, 1,  0,  0, 1, 0]  # C11 (k=1,i=1)
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct 4x4 decomposition from 2x2 components
    # Indices:
    # i, j, k in 0..3
    # i = 2*ih + il
    # j = 2*jh + jl
    # k = 2*kh + kl
    
    for c1 in range(7):
        for c2 in range(7):
            term_idx = c1 * 7 + c2
            
            # Fill U
            for ih in range(2):
                for jh in range(2):
                    u2_1 = U2[ih * 2 + jh][c1]
                    if u2_1 == 0: continue
                    for il in range(2):
                        for jl in range(2):
                            u2_2 = U2[il * 2 + jl][c2]
                            if u2_2 == 0: continue
                            val = u2_1 * u2_2
                            if val != 0:
                                i = 2 * ih + il
                                j = 2 * jh + jl
                                U[i * 4 + j][term_idx] = val
                                
            # Fill V
            for jh in range(2):
                for kh in range(2):
                    v2_1 = V2[jh * 2 + kh][c1]
                    if v2_1 == 0: continue
                    for jl in range(2):
                        for kl in range(2):
                            v2_2 = V2[jl * 2 + kl][c2]
                            if v2_2 == 0: continue
                            val = v2_1 * v2_2
                            if val != 0:
                                j = 2 * jh + jl
                                k = 2 * kh + kl
                                V[j * 4 + k][term_idx] = val
                                
            # Fill W
            # W index is k*n + i = k*4 + i
            for ih in range(2):
                for kh in range(2):
                    # W2 index is k*2 + i
                    w2_1 = W2[kh * 2 + ih][c1]
                    if w2_1 == 0: continue
                    for il in range(2):
                        for kl in range(2):
                            w2_2 = W2[kl * 2 + il][c2]
                            if w2_2 == 0: continue
                            val = w2_1 * w2_2
                            if val != 0:
                                i = 2 * ih + il
                                k = 2 * kh + kl
                                W[k * 4 + i][term_idx] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
