import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(r_a, c_a, r_b, c_b, r_c, c_c, rank_start):
        # A indices
        a00 = r_a * m + c_a
        a01 = r_a * m + c_a + 1
        a10 = (r_a + 1) * m + c_a
        a11 = (r_a + 1) * m + c_a + 1
        
        # B indices
        b00 = r_b * p + c_b
        b01 = r_b * p + c_b + 1
        b10 = (r_b + 1) * p + c_b
        b11 = (r_b + 1) * p + c_b + 1
        
        # C indices (W[k*n+i])
        c00 = c_c * n + r_c
        c10 = c_c * n + (r_c + 1)
        c01 = (c_c + 1) * n + r_c
        c11 = (c_c + 1) * n + (r_c + 1)
        
        # P1: (a00+a11)(b00+b11)
        U[a00][rank_start] += 1
        U[a11][rank_start] += 1
        V[b00][rank_start] += 1
        V[b11][rank_start] += 1
        W[c00][rank_start] += 1
        W[c11][rank_start] += 1
        
        # P2: (a10+a11)b00
        U[a10][rank_start+1] += 1
        U[a11][rank_start+1] += 1
        V[b00][rank_start+1] += 1
        W[c10][rank_start+1] += 1
        
        # P3: a00(b01-b11)
        U[a00][rank_start+2] += 1
        V[b01][rank_start+2] += 1
        V[b11][rank_start+2] -= 1
        W[c01][rank_start+2] += 1
        
        # P4: a11(b10-b00)
        U[a11][rank_start+3] += 1
        V[b10][rank_start+3] += 1
        V[b00][rank_start+3] -= 1
        W[c00][rank_start+3] += 1
        W[c10][rank_start+3] += 1
        
        # P5: (a00+a01)b11
        U[a00][rank_start+4] += 1
        U[a01][rank_start+4] += 1
        V[b11][rank_start+4] += 1
        W[c00][rank_start+4] -= 1
        W[c01][rank_start+4] += 1
        
        # P6: (a10-a00)(b00+b01)
        U[a10][rank_start+5] += 1
        U[a00][rank_start+5] -= 1
        V[b00][rank_start+5] += 1
        V[b01][rank_start+5] += 1
        W[c11][rank_start+5] += 1
        
        # P7: (a01-a11)(b10+b11)
        U[a01][rank_start+6] += 1
        U[a11][rank_start+6] -= 1
        V[b10][rank_start+6] += 1
        V[b11][rank_start+6] += 1
        W[c00][rank_start+6] += 1
        
    def add_schoolbook_1x4x4(r_a, c_a, r_b, c_b, r_c, c_c, rank_start):
        # A is 1x4, B is 4x4, C is 1x4
        for j in range(4):
            for k in range(4):
                u_idx = r_a * m + (c_a + j)
                v_idx = (r_b + j) * p + (c_b + k)
                w_idx = (c_c + k) * n + (r_c)
                
                U[u_idx][rank_start] += 1
                V[v_idx][rank_start] += 1
                W[w_idx][rank_start] += 1
                rank_start += 1
        return rank_start

    # Block 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    # Block 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2, 7)
    # Block 2a: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0, 14)
    # Block 2b: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # Block 3: A[2:3, 0:4] * B[0:4, 0:4] -> C[2:3, 0:4]
    current_rank = 28
    current_rank = add_schoolbook_1x4x4(2, 0, 0, 0, 2, 0, current_rank)
    
    return {"U": U, "V": V, "W": W}
