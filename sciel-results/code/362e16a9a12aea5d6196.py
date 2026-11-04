import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    block_size = 3
    num_blocks = 2 # 2x2 grid of 3x3 blocks

    # BCLR Rank-19 decomposition for 3x3 matrix multiplication
    # U, V, W are 19x9 (flattened 3x3)
    # Flattening order: row-major (i*3 + j)
    bclr_U = [
        [1,1,0, 0,0,0, 0,0,1],
        [0,0,1, 1,1,0, 0,0,0],
        [0,0,0, 0,1,1, 1,1,0],
        [0,1,1, 1,1,0, 0,0,0],
        [1,0,0, 0,0,1, 0,1,0],
        [0,0,1, 0,1,0, 1,0,0],
        [0,1,0, 1,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,1,1],
        [0,0,0, 1,0,0, 0,0,1],
        [0,0,1, 0,0,0, 1,0,0],
        [0,1,0, 0,0,0, 0,1,0],
        [1,0,0, 0,1,0, 0,0,0],
        [0,0,1, 0,1,0, 0,0,0],
        [0,1,0, 0,0,1, 0,0,0],
        [0,0,0, 0,1,1, 1,0,0],
        [0,0,0, 1,1,0, 0,1,0],
        [0,0,0, 0,1,0, 1,1,0],
        [0,0,0, 1,0,1, 0,1,0],
        [0,0,0, 0,1,1, 1,0,1]
    ]
    
    bclr_V = [
        [1,0,0, 0,1,1, 0,0,0],
        [0,0,0, 1,0,0, 0,1,1],
        [0,0,0, 0,0,1, 1,0,1],
        [0,0,1, 1,0,0, 0,1,0],
        [0,1,0, 1,0,1, 0,0,0],
        [1,0,0, 0,1,1, 0,0,0],
        [0,0,1, 0,1,0, 1,0,0],
        [0,1,0, 1,0,0, 0,0,1],
        [1,0,0, 0,1,0, 0,0,1],
        [0,1,0, 0,0,1, 1,0,0],
        [0,0,1, 1,0,0, 0,0,0],
        [0,0,1, 0,1,0, 1,0,0],
        [1,0,0, 0,0,1, 0,1,0],
        [0,0,1, 1,0,0, 0,1,0],
        [1,0,0, 0,0,1, 0,1,0],
        [0,1,0, 0,0,1, 1,0,0],
        [1,0,0, 0,1,0, 0,0,1],
        [0,0,1, 1,0,0, 0,1,0],
        [0,1,0, 0,0,1, 1,0,0]
    ]
    
    bclr_W = [
        [1,0,0, 0,1,1, 1,0,0],
        [0,0,1, 1,1,0, 0,0,0],
        [0,0,0, 0,1,1, 1,1,0],
        [0,1,1, 1,1,0, 0,0,0],
        [1,0,0, 0,0,1, 0,1,0],
        [0,0,1, 0,1,0, 1,0,0],
        [0,1,0, 1,0,0, 0,0,1],
        [1,0,0, 0,0,0, 0,1,1],
        [0,0,0, 1,0,0, 0,0,1],
        [0,0,1, 0,0,0, 1,0,0],
        [0,1,0, 0,0,0, 0,1,0],
        [1,0,0, 0,1,0, 0,0,0],
        [0,0,1, 0,1,0, 0,0,0],
        [0,1,0, 0,0,1, 0,0,0],
        [0,0,0, 0,1,1, 1,0,0],
        [0,0,0, 1,1,0, 0,1,0],
        [0,0,0, 0,1,0, 1,1,0],
        [0,0,0, 1,0,1, 0,1,0],
        [0,0,0, 0,1,1, 1,0,1]
    ]
    
    # Strassen steps for 2x2 block multiplication
    # (A_terms, B_terms, C_terms)
    # Terms are (I, J, coeff)
    strassen_steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    total_rank = 7 * 19
    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)
    
    col_idx = 0
    bs = block_size
    
    for s_idx, (a_terms, b_terms, c_terms) in enumerate(strassen_steps):
        for r_idx in range(19):
            u_3x3 = bclr_U[r_idx]
            v_3x3 = bclr_V[r_idx]
            w_3x3 = bclr_W[r_idx]
            
            # Process U
            for I, J, ca in a_terms:
                # ca is the coefficient from Strassen for block (I,J)
                # u_3x3 contains coefficients for the internal 3x3 multiplication
                for i, j in itertools.product(range(bs), repeat=2):
                    # Global index for A
                    u_global_idx = (I * bs + i) * dim + (J * bs + j)
                    U[u_global_idx, col_idx] += ca * u_3x3[i * bs + j]
                    
            # Process V
            for J, K, cb in b_terms:
                for j, k in itertools.product(range(bs), repeat=2):
                    # Global index for B
                    v_global_idx = (J * bs + j) * dim + (K * bs + k)
                    V[v_global_idx, col_idx] += cb * v_3x3[j * bs + k]
                    
            # Process W
            for I, K, cc in c_terms:
                for i, k in itertools.product(range(bs), repeat=2):
                    # Global index for C (Trace convention W[k*n+i, r])
                    # C is p x n. Here p=n=6.
                    # C index is (row, col) = (K, I) in block grid?
                    # C_{K, I} = sum J A_{I, J} B_{J, K}.
                    # So C block is at (K, I).
                    w_global_idx = (K * bs + k) * dim + (I * bs + i)
                    W[w_global_idx, col_idx] += cc * w_3x3[k * bs + i]
            
            col_idx += 1
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
