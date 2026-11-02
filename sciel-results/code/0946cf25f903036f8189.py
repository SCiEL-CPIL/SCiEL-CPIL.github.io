"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 8 blocks of size 2x2x2.
    # Each block has rank 7 (Strassen).
    # Total rank 56.
    
    num_blocks = p // 2 # 16 // 2 = 8
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)] # 4 x 56
    V = [[0] * total_rank for _ in range(m * p)] # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)] # 32 x 56
    
    # Strassen terms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b01 + b11)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define U, V, W coefficients for each term
    # U: (i, j) coefficients
    # V: (j, k) coefficients (local 2x2)
    # W: (k, i) coefficients (local 2x2)
    
    terms = []
    
    # Term 1
    # U: a00=1, a11=1
    u1 = {(0,0): 1, (1,1): 1}
    # V: b00=1, b11=1
    v1 = {(0,0): 1, (1,1): 1}
    # W: c00+=1, c11+=1
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # Term 2
    # U: a10=1, a11=1
    u2 = {(1,0): 1, (1,1): 1}
    # V: b00=1
    v2 = {(0,0): 1}
    # W: c10+=1
    w2 = {(0,1): 1} # C[1, 0] corresponds to k=0, i=1? 
    # Wait, C index is (i, k). 
    # My W index is (k, i). 
    # C[1, 0] means i=1, k=0. So W index (0, 1).
    terms.append((u2, v2, w2))
    
    # Term 3
    # U: a00=1
    u3 = {(0,0): 1}
    # V: b10=1, b11=-1
    v3 = {(1,0): 1, (1,1): -1}
    # W: c01+=1
    # C[0, 1] means i=0, k=1. W index (1, 0).
    w3 = {(1,0): 1}
    terms.append((u3, v3, w3))
    
    # Term 4
    # U: a11=1
    u4 = {(1,1): 1}
    # V: b01=1, b11=1
    v4 = {(0,1): 1, (1,1): 1}
    # W: c00+=1, c10+=1
    # C[0, 0] -> i=0, k=0 -> W(0,0)
    # C[1, 0] -> i=1, k=0 -> W(0,1)
    w4 = {(0,0): 1, (0,1): 1}
    terms.append((u4, v4, w4))
    
    # Term 5
    # U: a00=1, a01=1
    u5 = {(0,0): 1, (0,1): 1}
    # V: b11=1
    v5 = {(1,1): 1}
    # W: c00-=1, c01+=1
    # C[0, 0] -> W(0,0)
    # C[0, 1] -> W(1,0)
    w5 = {(0,0): -1, (1,0): 1}
    terms.append((u5, v5, w5))
    
    # Term 6
    # U: a10=1, a00=-1
    u6 = {(1,0): 1, (0,0): -1}
    # V: b00=1, b01=1
    v6 = {(0,0): 1, (0,1): 1}
    # W: c11+=1
    # C[1, 1] -> i=1, k=1 -> W(1,1)
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # Term 7
    # U: a01=1, a11=-1
    u7 = {(0,1): 1, (1,1): -1}
    # V: b10=1, b11=1
    v7 = {(1,0): 1, (1,1): 1}
    # W: c00+=1
    # C[0, 0] -> W(0,0)
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    for s in range(num_blocks):
        # Local block s corresponds to B columns 2s, 2s+1
        # And C columns 2s, 2s+1
        for t in range(rank_per_block):
            col = s * rank_per_block + t
            u_dict, v_dict, w_dict = terms[t]
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j][col] += val
                
            # Fill V
            # Local j, k maps to global j, 2s + k
            for (j, k), val in v_dict.items():
                V[j * p + (2 * s + k)][col] += val
                
            # Fill W
            # Local k, i maps to global k, i
            # Wait, W index is (k, i).
            # Global k is 2s + k_local.
            # Global i is i_local.
            for (k, i), val in w_dict.items():
                W[(2 * s + k) * n + i][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
