"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    # A is 2x4, B is 4x8
    # Split A into two 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    # Split B into two 2x8 blocks: B0 (rows 0-1), B1 (rows 2-3)
    # Split each 2x8 block into four 2x2 blocks
    # Total 8 blocks of 2x2x2 multiplication
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen decomposition for 2x2x2
    # U, V, W vectors for the 7 terms
    # Local indices:
    # U: i in {0,1}, j in {0,1} -> idx i*2+j
    # V: j in {0,1}, k in {0,1} -> idx j*2+k
    # W: k in {0,1}, i in {0,1} -> idx k*2+i (Trace convention W[k*n+i])
    
    strassen_U = [
        [1, 0, 1, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [-1, 0, 1, 0],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, 0], # M2: C10
        [0, 0, 1, 0], # M3: C01
        [1, 1, 0, 0], # M4: C00, C10
        [-1, 0, 1, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    col = 0
    # Loop over A blocks (2) and B blocks (4)
    for a_block in range(2):
        for b_block in range(4):
            for t in range(7):
                # U indices
                # A block a_block corresponds to global j offset a_block * 2
                # Local i in {0,1}, local j in {0,1}
                for i in range(2):
                    for j in range(2):
                        u_val = strassen_U[t][i * 2 + j]
                        if u_val != 0:
                            global_i = i
                            global_j = a_block * 2 + j
                            U[global_i * m + global_j][col] += u_val
                            
                # V indices
                # B block b_block corresponds to global k offset b_block * 2
                # B row offset: a_block determines which half of B rows?
                # Wait, A is 2x4. A = [A0, A1].
                # B is 4x8. B = [B0; B1].
                # C = A0 B0 + A1 B1.
                # So if a_block=0, we use B0 (rows 0-1).
                # If a_block=1, we use B1 (rows 2-3).
                b_row_offset = a_block * 2
                b_col_offset = b_block * 2
                
                for j in range(2):
                    for k in range(2):
                        v_val = strassen_V[t][j * 2 + k]
                        if v_val != 0:
                            global_j = b_row_offset + j
                            global_k = b_col_offset + k
                            V[global_j * p + global_k][col] += v_val
                            
                # W indices
                # Output C is 2x8.
                # Local i in {0,1}, local k in {0,1}
                for k in range(2):
                    for i in range(2):
                        w_val = strassen_W[t][k * 2 + i]
                        if w_val != 0:
                            global_i = i
                            global_k = b_block * 2 + k
                            W[global_k * n + global_i][col] += w_val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
