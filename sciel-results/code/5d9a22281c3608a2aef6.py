import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_to_U(row, col, val):
        U[row][col] += val
        
    def add_to_V(row, col, val):
        V[row][col] += val
        
    def add_to_W(row, col, val):
        W[row][col] += val
        
    low_level_steps = [
        {
            'u': [[1, 0], [0, 1]],
            'v': [[1, 0], [0, 1]],
            'w': [[1, 0], [0, 1]]
        },
        {
            'u': [[0, 0], [1, 1]],
            'v': [[1, 0], [0, 0]],
            'w': [[0, 0], [1, 0]]
        },
        {
            'u': [[1, 0], [0, 0]],
            'v': [[0, 1], [0, -1]],
            'w': [[0, 1], [0, 0]]
        },
        {
            'u': [[0, 0], [0, 1]],
            'v': [[-1, 0], [1, 0]],
            'w': [[1, 0], [1, 0]]
        },
        {
            'u': [[1, 1], [0, 0]],
            'v': [[0, 0], [0, 1]],
            'w': [[-1, 1], [0, 0]]
        },
        {
            'u': [[-1, 0], [1, 0]],
            'v': [[1, 1], [0, 0]],
            'w': [[0, 0], [0, 1]]
        },
        {
            'u': [[0, 1], [-1, 0]],
            'v': [[0, 0], [1, 1]],
            'w': [[1, 0], [0, 0]]
        }
    ]
    
    def make_coeffs_from_blocks(blocks):
        res = [[{} for _ in range(2)] for _ in range(2)]
        for br, bc, val in blocks:
            for r in range(2):
                for c in range(2):
                    i = br*2 + r
                    j = bc*2 + c
                    idx = i*4 + j
                    if idx not in res[r][c]:
                        res[r][c][idx] = 0
                    res[r][c][idx] += val
        return res

    high_level_X = [
        make_coeffs_from_blocks([(0,0,1), (1,1,1)]),
        make_coeffs_from_blocks([(1,0,1), (1,1,1)]),
        make_coeffs_from_blocks([(0,0,1)]),
        make_coeffs_from_blocks([(1,1,1)]),
        make_coeffs_from_blocks([(0,0,1), (0,1,1)]),
        make_coeffs_from_blocks([(1,0,1), (0,0,-1)]),
        make_coeffs_from_blocks([(0,1,1), (1,1,-1)])
    ]
    
    high_level_Y = [
        make_coeffs_from_blocks([(0,0,1), (1,1,1)]),
        make_coeffs_from_blocks([(0,0,1)]),
        make_coeffs_from_blocks([(0,1,1), (1,1,-1)]),
        make_coeffs_from_blocks([(1,0,1), (0,0,-1)]),
        make_coeffs_from_blocks([(1,1,1)]),
        make_coeffs_from_blocks([(0,0,1), (0,1,1)]),
        make_coeffs_from_blocks([(1,0,1), (1,1,1)])
    ]
    
    block_to_c = [
        [(0,0,1), (1,1,1)],
        [(1,0,1)],
        [(0,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    for m in range(7):
        for p_idx in range(7):
            col_idx = m*7 + p_idx
            step = low_level_steps[p_idx]
            
            for r in range(2):
                for c in range(2):
                    u_val = step['u'][r][c]
                    if u_val != 0:
                        for a_idx, a_coeff in high_level_X[m][r][c].items():
                            add_to_U(a_idx, col_idx, u_val * a_coeff)
                            
                    v_val = step['v'][r][c]
                    if v_val != 0:
                        for b_idx, b_coeff in high_level_Y[m][r][c].items():
                            add_to_V(b_idx, col_idx, v_val * b_coeff)
                            
            for r in range(2):
                for c in range(2):
                    w_val = step['w'][r][c]
                    if w_val != 0:
                        for c_br, c_bc, sign in block_to_c[m]:
                            i = c_br*2 + r
                            k = c_bc*2 + c
                            # W index is k*n + i
                            c_idx = k * n + i
                            add_to_W(c_idx, col_idx, w_val * sign)
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
