import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank
    # 4x4x4 part: 49
    # 4x4x1 part: 16
    # 4x1x4 part: 16
    # 4x1x1 part: 4
    rank = 49 + 16 + 16 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get 2x2x2 Strassen factors
    def get_strassen_2x2_factors():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Term 0: (a00+a11)(b00+b11)
        u[0][0]=1; u[3][0]=1
        v[0][0]=1; v[3][0]=1
        w[0][0]=1; w[3][0]=1
        
        # Term 1: (a10+a11)b00
        u[2][1]=1; u[3][1]=1
        v[0][1]=1
        w[2][1]=1; w[3][1]=-1
        
        # Term 2: a00(b01-b11)
        u[0][2]=1
        v[1][2]=1; v[3][2]=-1
        w[1][2]=1; w[3][2]=1
        
        # Term 3: a11(b10-b00)
        u[3][3]=1
        v[2][3]=1; v[0][3]=-1
        w[0][3]=1; w[2][3]=1
        
        # Term 4: (a00+a01)b11
        u[0][4]=1; u[1][4]=1
        v[3][4]=1
        w[0][4]=-1; w[1][4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        u[0][5]=-1; u[2][5]=1
        v[0][5]=1; v[1][5]=1
        w[3][5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        u[1][6]=1; u[3][6]=-1
        v[2][6]=1; v[3][6]=1
        w[0][6]=1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            u_vec = [s_u[r][inner_k] for r in range(4)]
            v_vec = [s_v[r][inner_k] for r in range(4)]
            w_vec = [s_w[r][inner_k] for r in range(4)]
            
            col_idx = start_col + inner_k
            
            for (a_br, a_bc), a_coeff in a_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1 part: A_L (4x4) * B_LR (4x1) -> C (4x1)
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    # 4x1x4 part: A_R (4x1) * B_RL (1x4) -> C (4x4)
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
            
    # 4x1x1 part: A_R (4x1) * B_RR (1x1) -> C (4x1)
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
