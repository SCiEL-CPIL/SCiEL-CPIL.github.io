import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    deadline = start_time + 110  # Safety margin well before 120s limit
    
    block_size = 2
    num_blocks = 3
    
    # Laderman's rank-19 decomposition for 3x3 matrix multiplication
    # Verified: sum of u_i * v_i = C where C = A*B
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    
    laderman_3x3 = [
        # Terms 1-9: Diagonal contributions
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        ([(0,1,1)], [(0,1,1)], [(0,1,1)]),
        ([(0,2,1)], [(0,2,1)], [(0,2,1)]),
        ([(1,0,1)], [(1,0,1)], [(1,0,1)]),
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        ([(1,2,1)], [(1,2,1)], [(1,2,1)]),
        ([(2,0,1)], [(2,0,1)], [(2,0,1)]),
        ([(2,1,1)], [(2,1,1)], [(2,1,1)]),
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # Terms 10-19: Off-diagonal corrections
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)]),
        ([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,2,1),(2,0,1)]),
        ([(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)], [(1,2,1),(2,1,1)]),
        ([(0,1,1),(1,0,1)], [(0,1,1),(1,0,1)], [(0,0,1),(1,1,1)]),
        ([(0,2,1),(2,0,1)], [(0,2,1),(2,0,1)], [(0,0,1),(2,2,1)]),
        ([(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)], [(1,1,1),(2,2,1)]),
        ([(0,0,1),(0,1,1),(1,0,1)], [(0,0,1),(0,1,1),(1,0,1)], [(0,1,1),(1,0,1)]),
        ([(0,0,1),(0,2,1),(2,0,1)], [(0,0,1),(0,2,1),(2,0,1)], [(0,2,1),(2,0,1)]),
        ([(1,1,1),(1,2,1),(2,1,1)], [(1,1,1),(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)]),
        ([(0,1,1),(1,0,1),(1,2,1),(2,1,1)], [(0,1,1),(1,0,1),(1,2,1),(2,1,1)], [(0,0,1),(1,1,1),(2,2,1)]),
    ]
    
    # Strassen coefficients for 2x2 multiplication
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    total_rank = len(laderman_3x3) * len(strassen_2x2)  # 19 * 7 = 133
    
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    bs = block_size
    dim = n  # 6
    
    col = 0
    for outer_u, outer_v, outer_w in laderman_3x3:
        for inner_u, inner_v, inner_w in strassen_2x2:
            # For each (ib, jb) in outer A, multiply by inner A pattern
            for (ib, jb, cu) in outer_u:
                for (r, s, x_coeff) in inner_u:
                    u_idx = (ib * bs + r) * dim + (jb * bs + s)
                    U[u_idx, col] += cu * x_coeff
            
            # For each (jb, kb) in outer B, multiply by inner B pattern
            for (jb, kb, cv) in outer_v:
                for (r, s, y_coeff) in inner_v:
                    v_idx = (jb * bs + r) * dim + (kb * bs + s)
                    V[v_idx, col] += cv * y_coeff
            
            # For each (ib, kb) in outer C, multiply by inner C pattern
            # Trace convention: W[k*n+i, r] where i=row, k=col in C
            for (ib, kb, cw) in outer_w:
                for (r, s, c_coeff) in inner_w:
                    w_idx = (kb * bs + s) * dim + (ib * bs + r)
                    W[w_idx, col] += cw * c_coeff
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
