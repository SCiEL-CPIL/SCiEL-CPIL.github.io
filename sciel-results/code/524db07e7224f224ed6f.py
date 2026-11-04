import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _check_time():
    if _time_remaining() < 5.0:
        raise TimeoutError("Time limit approaching")

# Precomputed Strassen 2x2x2 terms: (x_dict, y_dict, c_dict)
_STRASSEN_2x2 = [
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]

# Strassen 4x4x4 block combinations
_STRASSEN_4x4 = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0,0): 1, (1,1): 1}),
    ({2: 1, 3: 1}, {0: 1}, {(1,0): 1, (1,1): -1}),
    ({0: 1}, {1: 1, 3: -1}, {(0,1): 1, (1,1): 1}),
    ({3: 1}, {2: 1, 0: -1}, {(0,0): 1, (1,0): 1}),
    ({0: 1, 1: 1}, {3: 1}, {(0,0): -1, (0,1): 1}),
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1,1): 1}),
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0,0): 1}),
]

def _build_4x4x4(U, V, W, start_rank, n, m, p):
    rank = start_rank
    for m_idx in range(7):
        x_def, y_def, c_block = _STRASSEN_4x4[m_idx]
        for t_idx in range(7):
            x_sub, y_sub, c_sub = _STRASSEN_2x2[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi, bj = b // 2, b % 2
                    i, j = 2*bi + li, 2*bj + lj
                    u_coeffs[i*m + j] += c_x * coeff_block
            
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi, bj = b // 2, b % 2
                    i, j = 2*bi + li, 2*bj + lj
                    v_coeffs[i*p + j] += c_y * coeff_block
            
            for i in range(n*m):
                U[i][rank] = u_coeffs[i]
            for j in range(m*p):
                V[j][rank] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            for (li, lj), cm_c in c_sub.items():
                for (br, bc), cb_c in c_block.items():
                    i_final = 2*br + li
                    k_final = 2*bc + lj
                    w_coeffs[k_final*n + i_final] += cm_c * cb_c
            
            for i in range(p*n):
                W[i][rank] = w_coeffs[i]
            rank += 1
    return rank

def _build_2x2x3_block(U, V, W, start_rank, n, m, p, ar, ac, br, cr):
    rank = start_rank
    for t_idx in range(7):
        x_sub, y_sub, c_sub = _STRASSEN_2x2[t_idx]
        
        u_coeffs = [0.0] * (n*m)
        v_coeffs = [0.0] * (m*p)
        
        for (li, lj), c_x in x_sub.items():
            i, j = 2*ar + li, 2*ac + lj
            u_coeffs[i*m + j] += c_x
        
        for (li, lj), c_y in y_sub.items():
            i, j = 2*br + li, 4 + lj
            v_coeffs[i*p + j] += c_y
        
        for i in range(n*m):
            U[i][rank] = u_coeffs[i]
        for j in range(m*p):
            V[j][rank] = v_coeffs[j]
        
        w_coeffs = [0.0] * (p*n)
        for (li, lj), cm_c in c_sub.items():
            i_final = 2*cr + li
            k_final = 4 + lj
            w_coeffs[k_final*n + i_final] += cm_c
        
        for i in range(p*n):
            W[i][rank] = w_coeffs[i]
        rank += 1
    
    schoolbook = [
        ((0,0), (0,0), (0,0)),
        ((0,1), (1,0), (0,0)),
        ((1,0), (0,0), (1,0)),
        ((1,1), (1,0), (1,0)),
    ]
    for (x_pos, y_pos, c_pos) in schoolbook:
        u_coeffs = [0.0] * (n*m)
        v_coeffs = [0.0] * (m*p)
        u_coeffs[(2*ar + x_pos[0])*m + (2*ac + x_pos[1])] = 1.0
        v_coeffs[(2*br + y_pos[0])*p + 6] = 1.0
        
        for i in range(n*m):
            U[i][rank] = u_coeffs[i]
        for j in range(m*p):
            V[j][rank] = v_coeffs[j]
        
        w_coeffs = [0.0] * (p*n)
        w_coeffs[6*n + (2*cr + c_pos[0])] = 1.0
        for i in range(p*n):
            W[i][rank] = w_coeffs[i]
        rank += 1
    
    return rank

def construct():
    n, m, p = DIMENSIONS
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    _check_time()
    current_rank = _build_4x4x4(U, V, W, 0, n, m, p)
    
    _check_time()
    current_rank = _build_2x2x3_block(U, V, W, current_rank, n, m, p, 0, 0, 0, 0)
    current_rank = _build_2x2x3_block(U, V, W, current_rank, n, m, p, 0, 1, 1, 0)
    current_rank = _build_2x2x3_block(U, V, W, current_rank, n, m, p, 1, 0, 0, 1)
    current_rank = _build_2x2x3_block(U, V, W, current_rank, n, m, p, 1, 1, 1, 1)
    
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
