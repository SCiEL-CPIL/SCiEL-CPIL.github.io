import numpy as np

def construct():
    """
    Decompose 2x2 * 2x10 multiplication into five independent 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35
    
    # Initialize matrices as zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen rank-7 decomposition for 2x2 matrices
    # We define the 7 terms as tuples of (U_dict, V_dict, W_dict)
    # U_dict: {(i, j): coeff} for A[i, j]
    # V_dict: {(j, k): coeff} for B[j, k]
    # W_dict: {(k, i): coeff} for C[i, k] (note W index is k*n + i)
    
    strassen_terms = [
        # 0: (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
        # 1: (c+d)e
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(0,1): 1, (1,1): -1}),
        # 2: a(f-h)
        ({(0,0): 1}, 
         {(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}),
        # 3: d(g-e)
        ({(1,1): 1}, 
         {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}),
        # 4: (a+b)h
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (1,0): 1}),
        # 5: (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        # 6: (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1})
    ]
    
    for block_idx in range(num_blocks):
        base_col = block_idx * 7
        for t_idx in range(7):
            col = base_col + t_idx
            u_dict, v_dict, w_dict = strassen_terms[t_idx]
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j][col] = val
                
            # Fill V
            for (j, k), val in v_dict.items():
                global_k = 2 * block_idx + k
                V[j * p + global_k][col] = val
                
            # Fill W
            for (k, i), val in w_dict.items():
                global_k = 2 * block_idx + k
                W[global_k * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
