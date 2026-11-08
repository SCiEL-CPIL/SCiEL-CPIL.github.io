import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS
    rank = 49 + 16 + 16 + 4
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def get_strassen_2x2_factors():
        u = np.zeros((4, 7), dtype=int)
        v = np.zeros((4, 7), dtype=int)
        w = np.zeros((4, 7), dtype=int)
        
        # Term 0: (a00+a11)(b00+b11)
        u[0,0]=1; u[3,0]=1
        v[0,0]=1; v[3,0]=1
        w[0,0]=1; w[3,0]=1
        
        # Term 1: (a10+a11)b00
        u[2,1]=1; u[3,1]=1
        v[0,1]=1
        w[2,1]=1; w[3,1]=-1
        
        # Term 2: a00(b01-b11)
        u[0,2]=1
        v[1,2]=1; v[3,2]=-1
        w[1,2]=1; w[3,2]=1
        
        # Term 3: a11(b10-b00)
        u[3,3]=1
        v[2,3]=1; v[0,3]=-1
        w[0,3]=1; w[2,3]=1
        
        # Term 4: (a00+a01)b11
        u[0,4]=1; u[1,4]=1
        v[3,4]=1
        w[0,4]=-1; w[1,4]=1
        
        # Term 5: (a10-a00)(b00+b01)
        u[0,5]=-1; u[2,5]=1
        v[0,5]=1; v[1,5]=1
        w[3,5]=1
        
        # Term 6: (a01-a11)(b10+b11)
        u[1,6]=1; u[3,6]=-1
        v[2,6]=1; v[3,6]=1
        w[0,6]=1
        
        return u, v, w

    s_u, s_v, s_w = get_strassen_2x2_factors()
    
    def add_4x4_term(a_block_coeffs, b_block_coeffs, c_block_coeffs, start_col):
        for inner_k in range(7):
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
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
                            U[global_u_row, col_idx] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row, col_idx] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_block_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row, col_idx] += c_coeff * w_val

    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)
        
    col = 49
    
    # 4x4x1 part
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    # 4x1x4 part
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            
    # 4x1x1 part
    for i in range(4):
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
        
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
