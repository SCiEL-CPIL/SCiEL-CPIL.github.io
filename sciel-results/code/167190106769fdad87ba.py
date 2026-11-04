import numpy as np

def construct():
    n, m, p = 2, 3, 8
    # Dimensions
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Target rank: 1 (for 2x1x8) + 4 * 7 (for 2x2x2 Strassen) = 29
    rank = 29
    
    U = np.zeros((dim_U, rank), dtype=float)
    V = np.zeros((dim_V, rank), dtype=float)
    W = np.zeros((dim_W, rank), dtype=float)
    
    # Block 1: Handle j=2 (the 3rd dimension of A and B)
    # T[i, 2, k] = 1 for all i, k
    # Rank 1 decomposition:
    # U[i, 0] = 1 for i in {0, 1} (indices 2, 5 in flattened A)
    # V[2, k, 0] = 1 for all k (indices 16..23 in flattened B)
    # W[k, i, 0] = 1 for all k, i (indices 0..15 in flattened C)
    
    c = 0
    # U indices for i=0, j=2 is 0*3+2=2; i=1, j=2 is 1*3+2=5
    U[2, c] = 1
    U[5, c] = 1
    
    # V indices for j=2, k=0..7 is 2*8+k = 16+k
    for k in range(p):
        V[16 + k, c] = 1
        
    # W indices for k=0..7, i=0..1 is k*2+i
    for k in range(p):
        for i in range(n):
            W[k * n + i, c] = 1
            
    # Block 2: Handle j in {0, 1} using 4 Strassen blocks for k groups
    # Strassen coefficients for 2x2x2
    # s=1..7
    
    # Linear forms on A (local i', j')
    # x = i'*3 + j'
    A_coeffs = {
        1: {0: {0: 1, 1: 0}, 1: {0: 1, 1: 1}}, # s=1: A00+A11. Wait.
        # Let's just use lists of (local_i, local_j, coeff)
    }
    
    # Define Strassen steps as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i', j', val)
    # v_coeffs: list of (j', k', val)
    # w_coeffs: list of (i', k', val)
    
    steps = []
    
    # s=1
    steps.append({
        'u': [(0,0,1), (1,1,1)],
        'v': [(0,0,1), (1,1,1)],
        'w': [(0,0,1), (1,1,1)] # P1 contributes to C00 and C11?
        # C00 = P1 + P4 - P5 + P7
        # C11 = P1 - P2 + P3 + P6
        # So w should reflect gamma.
    })
    
    # It is better to define the contribution to W directly based on gamma.
    # For each step s, W_{z,c} = gamma_{i,k,s} if z matches (k,i).
    
    # Let's re-structure:
    # For each s, we know u_coeffs and v_coeffs.
    # We also know which (i', k') pairs get which coefficient from P_s.
    
    strassen_data = [
        # s=1
        {
            'u': [(0,0,1), (1,1,1)],
            'v': [(0,0,1), (1,1,1)],
            'w': [(0,0,1), (1,1,1)] # C00+=1, C11+=1
        },
        # s=2
        {
            'u': [(1,0,1), (1,1,1)],
            'v': [(0,0,1)],
            'w': [(1,0,1), (1,1,-1)] # C10+=1, C11-=1
        },
        # s=3
        {
            'u': [(0,0,1)],
            'v': [(1,0,1), (1,1,-1)],
            'w': [(0,1,1), (1,1,1)] # C01+=1, C11+=1
        },
        # s=4
        {
            'u': [(1,1,1)],
            'v': [(0,1,1), (0,0,-1)],
            'w': [(0,0,1), (1,0,1)] # C00+=1, C10+=1
        },
        # s=5
        {
            'u': [(0,0,1), (1,0,1)],
            'v': [(1,1,1)],
            'w': [(0,0,-1), (0,1,1)] # C00-=1, C01+=1
        },
        # s=6
        {
            'u': [(1,1,1), (0,0,-1)],
            'v': [(0,0,1), (1,0,1)],
            'w': [(1,1,1)] # C11+=1
        },
        # s=7
        {
            'u': [(0,1,1), (1,1,-1)],
            'v': [(1,1,1), (0,1,1)],
            'w': [(0,0,1)] # C00+=1
        }
    ]
    
    for m in range(4):
        base_k = m * 2
        for s_idx, data in enumerate(strassen_data):
            c = 1 + m * 7 + s_idx
            
            # Fill U
            for li, lj, val in data['u']:
                global_x = li * m + lj
                U[global_x, c] = val
                
            # Fill V
            for lj, lk, val in data['v']:
                global_k = base_k + lk
                global_y = lj * p + global_k
                V[global_y, c] = val
                
            # Fill W
            for li, lk, val in data['w']:
                global_k = base_k + lk
                global_z = global_k * n + li
                W[global_z, c] = val
                
    return {"U": U, "V": V, "W": W}
