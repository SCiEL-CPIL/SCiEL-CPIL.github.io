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

    # Global matrices
    total_rank = 124
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    def add_block(u_local, v_local, w_local, 
                  a_r, a_c, a_h, a_w, 
                  b_r, b_c, b_h, b_w, 
                  c_r, c_c, c_h, c_w):
        nonlocal rank_offset
        local_rank = len(u_local[0])
        
        # Map U
        for i in range(a_h):
            for j in range(a_w):
                local_idx = i * a_w + j
                global_idx = (a_r + i) * m + (a_c + j)
                for r in range(local_rank):
                    U[global_idx][rank_offset + r] = u_local[local_idx][r]
                    
        # Map V
        for i in range(b_h):
            for j in range(b_w):
                local_idx = i * b_w + j
                global_idx = (b_r + i) * p + (b_c + j)
                for r in range(local_rank):
                    V[global_idx][rank_offset + r] = v_local[local_idx][r]
                    
        # Map W
        # Local W index is k * c_w + i ?
        # In schoolbook: w[k * n + i]. Here n is the number of columns of C?
        # C is c_h x c_w.
        # Local W has c_w * c_h rows?
        # In schoolbook(n, m, p), w has p * n rows.
        # Here, the "p" of the block is c_w, and "n" of the block is c_h.
        # So local W index is k * c_h + i?
        # Let's check schoolbook:
        # u[i * m + j]
        # v[j * p + k]
        # w[k * n + i]
        # So for block C (c_h x c_w), the local W index is k * c_h + i.
        # Wait, in schoolbook(n,m,p), n is rows of A/C, p is cols of B/C.
        # So C is n x p.
        # w index is k * n + i.
        # So for block C (c_h x c_w), c_h corresponds to n, c_w corresponds to p.
        # So local W index is k * c_h + i.
        
        for k in range(c_w):
            for i in range(c_h):
                local_idx = k * c_h + i
                global_idx = (c_r + k) * n + (c_c + i)
                for r in range(local_rank):
                    W[global_idx][rank_offset + r] = w_local[local_idx][r]
                    
        rank_offset += local_rank

    # 1. A11 (2x2) * B11 (2x2) -> C11 (2x2)
    u1, v1, w1 = strassen_2x2()
    add_block(u1, v1, w1, 
              0, 0, 2, 2, 
              0, 0, 2, 2, 
              0, 0, 2, 2)
              
    # 2. A12 (2x3) * B21 (3x2) -> C11 (2x2)
    u2, v2, w2 = schoolbook(2, 3, 2)
    add_block(u2, v2, w2, 
              0, 2, 2, 3, 
              2, 0, 3, 2, 
              0, 0, 2, 2)
              
    # 3. A11 (2x2) * B12 (2x3) -> C12 (2x3)
    u3, v3, w3 = schoolbook(2, 2, 3)
    add_block(u3, v3, w3, 
              0, 0, 2, 2, 
              0, 2, 2, 3, 
              0, 2, 2, 3)
              
    # 4. A12 (2x3) * B22 (3x3) -> C12 (2x3)
    u4, v4, w4 = schoolbook(2, 3, 3)
    add_block(u4, v4, w4, 
              0, 2, 2, 3, 
              2, 2, 3, 3, 
              0, 2, 2, 3)
              
    # 5. A21 (3x2) * B11 (2x2) -> C21 (3x2)
    u5, v5, w5 = schoolbook(3, 2, 2)
    add_block(u5, v5, w5, 
              2, 0, 3, 2, 
              0, 0, 2, 2, 
              2, 0, 3, 2)
              
    # 6. A22 (3x3) * B21 (3x2) -> C21 (3x2)
    u6, v6, w6 = schoolbook(3, 3, 2)
    add_block(u6, v6, w6, 
              2, 2, 3, 3, 
              2, 0, 3, 2, 
              2, 0, 3, 2)
              
    # 7. A21 (3x2) * B12 (2x3) -> C22 (3x3)
    u7, v7, w7 = schoolbook(3, 2, 3)
    add_block(u7, v7, w7, 
              2, 0, 3, 2, 
              0, 2, 2, 3, 
              2, 2, 3, 3)
              
    # 8. A22 (3x3) * B22 (3x3) -> C22 (3x3)
    u8, v8, w8 = schoolbook(3, 3, 3)
    add_block(u8, v8, w8, 
              2, 2, 3, 3, 
              2, 2, 3, 3, 
              2, 2, 3, 3)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
