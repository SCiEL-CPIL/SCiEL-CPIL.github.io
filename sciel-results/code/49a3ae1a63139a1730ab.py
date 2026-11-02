import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # p=14, so 7 blocks of 2 columns
    num_blocks = p // 2
    rank = 7 * num_blocks
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Terms: (u_dict, v_dict, w_dict) with local indices
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),              # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),              # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),              # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),              # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),              # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),              # M7
    ]
    
    col = 0
    for b in range(num_blocks):
        # Global offsets for V and W
        # V local index l corresponds to:
        # 0 -> B00 -> 14*0 + 2*b
        # 1 -> B01 -> 14*0 + 2*b + 1
        # 2 -> B10 -> 14*1 + 2*b
        # 3 -> B11 -> 14*1 + 2*b + 1
        
        # W local index l corresponds to:
        # 0 -> C00 -> 2*(2*b) + 0
        # 1 -> C01 -> 2*(2*b+1) + 0
        # 2 -> C10 -> 2*(2*b) + 1
        # 3 -> C11 -> 2*(2*b+1) + 1
        
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            # Fill U
            for idx, val in u_dict.items():
                U[idx][col] += val
            # Fill V
            for l_idx, val in v_dict.items():
                global_v_idx = v_offsets[l_idx]
                V[global_v_idx][col] += val
            # Fill W
            for l_idx, val in w_dict.items():
                global_w_idx = w_offsets[l_idx]
                W[global_w_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
