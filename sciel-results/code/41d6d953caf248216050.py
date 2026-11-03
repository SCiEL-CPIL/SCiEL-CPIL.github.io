import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a schoolbook decomposition for n x m by m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        u = [[0] * rank for _ in range(n * m)]
        v = [[0] * rank for _ in range(m * p)]
        w = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u[i * m + j][col] = 1
                    v[j * p + k][col] = 1
                    w[k * n + i][col] = 1
                    col += 1
        return u, v, w

    # Helper to create a Strassen decomposition for 2x2
    def strassen_2x2():
        # Returns U, V, W for 2x2 multiplication
        # Rank 7
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # m1 = (a+d)(e+h)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1
        w[3][0] = 1
        
        # m2 = (c+d)e
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1
        w[3][1] = -1
        
        # m3 = a(f-h)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1
        w[3][2] = 1
        
        # m4 = d(g-e)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1
        w[2][3] = 1
        
        # m5 = (a+b)h
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1
        w[1][4] = 1
        
        # m6 = (c-d)(f+h)
        u[2][5] = 1; u[3][5] = -1
        v[1][5] = 1; v[3][5] = 1
        w[3][5] = 1
        
        # m7 = (b-d)(e+g)
        u[1][6] = 1; u[3][6] = -1
        v[0][6] = 1; v[2][6] = 1
        w[0][6] = 1
        
        return u, v, w

    # Helper to combine decompositions
    # u_local: (n*m, rank_local)
    # v_local: (m*p, rank_local)
    # w_local: (p*n, rank_local)
    # u_map: list of (global_row, coeff) for each local row
    # v_map: list of (global_col, coeff) for each local col
    # w_map: list of (global_out, coeff) for each local out
    # rank_offset: offset in the global rank dimension
    def combine(u_global, v_global, w_global, u_local, v_local, w_local, u_map, v_map, w_map, rank_offset):
        rank_local = len(u_local[0])
        
        for i in range(len(u_local)):
            global_row, coeff = u_map[i]
            for r in range(rank_local):
                u_global[global_row][rank_offset + r] += coeff * u_local[i][r]
                
        for j in range(len(v_local)):
            global_col, coeff = v_map[j]
            for r in range(rank_local):
                v_global[global_col][rank_offset + r] += coeff * v_local[j][r]
                
        for k in range(len(w_local)):
            global_out, coeff = w_map[k]
            for r in range(rank_local):
                w_global[global_out][rank_offset + r] += coeff * w_local[k][r]

    # Construct 4x4 Strassen decomposition
    def strassen_4x4():
        # n=m=p=4
        # Split into 2x2 blocks
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        
        # We need 7 multiplications of 2x2 matrices.
        # Total rank = 7 * 7 = 49.
        
        u_global = [[0]*49 for _ in range(16)]
        v_global = [[0]*49 for _ in range(16)]
        w_global = [[0]*49 for _ in range(16)]
        
        # Local 2x2 Strassen decomposition
        u2, v2, w2 = strassen_2x2()
        
        # Indices for 2x2 blocks in 4x4 matrix (row-major)
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # M1 = (A11 + A22)(B11 + B22)
        # u_map: A11 (1), A22 (1)
        u_map_1 = [
            (0, 1), (1, 1), (4, 1), (5, 1),  # A11
            (10, 1), (11, 1), (14, 1), (15, 1) # A22
        ]
        # v_map: B11 (1), B22 (1)
        v_map_1 = [
            (0, 1), (1, 1), (4, 1), (5, 1),  # B11
            (10, 1), (11, 1), (14, 1), (15, 1) # B22
        ]
        # w_map: C11 (1), C22 (1)
        w_map_1 = [
            (0, 1), (1, 1), (4, 1), (5, 1),  # C11
            (10, 1), (11, 1), (14, 1), (15, 1) # C22
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_1, v_map_1, w_map_1, 0)
        
        # M2 = (A21 + A22)B11
        u_map_2 = [
            (8, 1), (9, 1), (12, 1), (13, 1),  # A21
            (10, 1), (11, 1), (14, 1), (15, 1) # A22
        ]
        v_map_2 = [
            (0, 1), (1, 1), (4, 1), (5, 1)     # B11
        ]
        w_map_2 = [
            (8, 1), (9, 1), (12, 1), (13, 1),  # C21
            (10, -1), (11, -1), (14, -1), (15, -1) # C22
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_2, v_map_2, w_map_2, 7)
        
        # M3 = A11(B12 - B22)
        u_map_3 = [
            (0, 1), (1, 1), (4, 1), (5, 1)     # A11
        ]
        v_map_3 = [
            (2, 1), (3, 1), (6, 1), (7, 1),    # B12
            (10, -1), (11, -1), (14, -1), (15, -1) # B22
        ]
        w_map_3 = [
            (2, 1), (3, 1), (6, 1), (7, 1),    # C12
            (10, 1), (11, 1), (14, 1), (15, 1) # C22
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_3, v_map_3, w_map_3, 14)
        
        # M4 = A22(B21 - B11)
        u_map_4 = [
            (10, 1), (11, 1), (14, 1), (15, 1) # A22
        ]
        v_map_4 = [
            (8, 1), (9, 1), (12, 1), (13, 1),  # B21
            (0, -1), (1, -1), (4, -1), (5, -1) # B11
        ]
        w_map_4 = [
            (0, 1), (1, 1), (4, 1), (5, 1),    # C11
            (8, 1), (9, 1), (12, 1), (13, 1)   # C21
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_4, v_map_4, w_map_4, 21)
        
        # M5 = (A11 + A12)B22
        u_map_5 = [
            (0, 1), (1, 1), (4, 1), (5, 1),    # A11
            (2, 1), (3, 1), (6, 1), (7, 1)     # A12
        ]
        v_map_5 = [
            (10, 1), (11, 1), (14, 1), (15, 1) # B22
        ]
        w_map_5 = [
            (0, -1), (1, -1), (4, -1), (5, -1), # C11
            (2, 1), (3, 1), (6, 1), (7, 1)      # C12
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_5, v_map_5, w_map_5, 28)
        
        # M6 = (A21 - A11)(B12 + B22)
        u_map_6 = [
            (8, 1), (9, 1), (12, 1), (13, 1),  # A21
            (0, -1), (1, -1), (4, -1), (5, -1) # A11
        ]
        v_map_6 = [
            (2, 1), (3, 1), (6, 1), (7, 1),    # B12
            (10, 1), (11, 1), (14, 1), (15, 1) # B22
        ]
        w_map_6 = [
            (10, 1), (11, 1), (14, 1), (15, 1) # C22
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_6, v_map_6, w_map_6, 35)
        
        # M7 = (A12 - A22)(B11 + B21)
        u_map_7 = [
            (2, 1), (3, 1), (6, 1), (7, 1),    # A12
            (10, -1), (11, -1), (14, -1), (15, -1) # A22
        ]
        v_map_7 = [
            (0, 1), (1, 1), (4, 1), (5, 1),    # B11
            (8, 1), (9, 1), (12, 1), (13, 1)   # B21
        ]
        w_map_7 = [
            (0, 1), (1, 1), (4, 1), (5, 1)     # C11
        ]
        combine(u_global, v_global, w_global, u2, v2, w2, u_map_7, v_map_7, w_map_7, 42)
        
        return u_global, v_global, w_global

    # Main construction: 5x5 = 4+1 split
    # A = [[A44, A41], [A14, A11]]
    # B = [[B44, B41], [B14, B11]]
    
    # 1. A44 B44 -> C44_part1 (Rank 49)
    u1, v1, w1 = strassen_4x4()
    
    # 2. A41 B14 -> C44_part2 (4x1 x 1x4 -> 4x4) (Rank 16)
    u2, v2, w2 = schoolbook(4, 1, 4)
    
    # 3. A44 B41 -> C41 (4x4 x 4x1 -> 4x1) (Rank 16)
    u3, v3, w3 = schoolbook(4, 4, 1)
    
    # 4. A41 B11 -> C41_part2 (4x1 x 1x1 -> 4x1) (Rank 4)
    u4, v4, w4 = schoolbook(4, 1, 1)
    
    # 5. A14 B44 -> C14 (1x4 x 4x4 -> 1x4) (Rank 16)
    u5, v5, w5 = schoolbook(1, 4, 4)
    
    # 6. A11 B14 -> C14_part2 (1x1 x 1x4 -> 1x4) (Rank 4)
    u6, v6, w6 = schoolbook(1, 1, 4)
    
    # 7. A14 B41 -> C11 (1x4 x 4x1 -> 1x1) (Rank 4)
    u7, v7, w7 = schoolbook(1, 4, 1)
    
    # 8. A11 B11 -> C11_part2 (1x1 x 1x1 -> 1x1) (Rank 1)
    u8, v8, w8 = schoolbook(1, 1, 1)
    
    total_rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    # Maps for 4x4 block (top-left)
    # A44: rows 0-3, cols 0-3
    u_map_44 = [i*5+j for i in range(4) for j in range(4)]
    v_map_44 = [i*5+j for i in range(4) for j in range(4)]
    w_map_44 = [i*5+j for i in range(4) for j in range(4)]
    
    # A41: rows 0-3, cols 4
    u_map_41 = [i*5+4 for i in range(4)]
    # B14: rows 4, cols 0-3
    v_map_14 = [4*5+j for j in range(4)]
    # C44: rows 0-3, cols 0-3
    w_map_44_out = [i*5+j for i in range(4) for j in range(4)]
    
    # A44: rows 0-3, cols 0-3
    u_map_44_in = [i*5+j for i in range(4) for j in range(4)]
    # B41: rows 0-3, cols 4
    v_map_41 = [i*5+4 for i in range(4)]
    # C41: rows 0-3, col 4
    w_map_41_out = [i*5+4 for i in range(4)]
    
    # A41: rows 0-3, col 4
    u_map_41_in = [i*5+4 for i in range(4)]
    # B11: row 4, col 4
    v_map_11 = [4*5+4]
    # C41: rows 0-3, col 4
    w_map_41_out2 = [i*5+4 for i in range(4)]
    
    # A14: row 4, cols 0-3
    u_map_14 = [4*5+j for j in range(4)]
    # B44: rows 0-3, cols 0-3
    v_map_44_in = [i*5+j for i in range(4) for j in range(4)]
    # C14: row 4, cols 0-3
    w_map_14_out = [4*5+j for j in range(4)]
    
    # A11: row 4, col 4
    u_map_11 = [4*5+4]
    # B14: row 4, cols 0-3
    v_map_14_in = [4*5+j for j in range(4)]
    # C14: row 4, cols 0-3
    w_map_14_out2 = [4*5+j for j in range(4)]
    
    # A14: row 4, cols 0-3
    u_map_14_in = [4*5+j for j in range(4)]
    # B41: rows 0-3, col 4
    v_map_41_in = [i*5+4 for i in range(4)]
    # C11: row 4, col 4
    w_map_11_out = [4*5+4]
    
    # A11: row 4, col 4
    u_map_11_in = [4*5+4]
    # B11: row 4, col 4
    v_map_11_in = [4*5+4]
    # C11: row 4, col 4
    w_map_11_out2 = [4*5+4]

    # Place 1: A44 B44
    combine(U, V, W, u1, v1, w1, 
            [(idx, 1) for idx in u_map_44], 
            [(idx, 1) for idx in v_map_44], 
            [(idx, 1) for idx in w_map_44], 0)
            
    # Place 2: A41 B14
    combine(U, V, W, u2, v2, w2, 
            [(idx, 1) for idx in u_map_41], 
            [(idx, 1) for idx in v_map_14], 
            [(idx, 1) for idx in w_map_44_out], 49)
            
    # Place 3: A44 B41
    combine(U, V, W, u3, v3, w3, 
            [(idx, 1) for idx in u_map_44_in], 
            [(idx, 1) for idx in v_map_41], 
            [(idx, 1) for idx in w_map_41_out], 65)
            
    # Place 4: A41 B11
    combine(U, V, W, u4, v4, w4, 
            [(idx, 1) for idx in u_map_41_in], 
            [(idx, 1) for idx in v_map_11], 
            [(idx, 1) for idx in w_map_41_out2], 81)
            
    # Place 5: A14 B44
    combine(U, V, W, u5, v5, w5, 
            [(idx, 1) for idx in u_map_14], 
            [(idx, 1) for idx in v_map_44_in], 
            [(idx, 1) for idx in w_map_14_out], 85)
            
    # Place 6: A11 B14
    combine(U, V, W, u6, v6, w6, 
            [(idx, 1) for idx in u_map_11], 
            [(idx, 1) for idx in v_map_14_in], 
            [(idx, 1) for idx in w_map_14_out2], 101)
            
    # Place 7: A14 B41
    combine(U, V, W, u7, v7, w7, 
            [(idx, 1) for idx in u_map_14_in], 
            [(idx, 1) for idx in v_map_41_in], 
            [(idx, 1) for idx in w_map_11_out], 105)
            
    # Place 8: A11 B11
    combine(U, V, W, u8, v8, w8, 
            [(idx, 1) for idx in u_map_11_in], 
            [(idx, 1) for idx in v_map_11_in], 
            [(idx, 1) for idx in w_map_11_out2], 109)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
