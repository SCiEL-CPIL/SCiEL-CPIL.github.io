import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # rank = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen coefficients
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:11, 1:12, 2:21, 3:22 (row-major? No, task uses k*n+i)
    # Task W index: k*n + i. C is p x n.
    # Strassen C is 2x2.
    # C[0,0] (i=0, k=0) -> index 0*2+0 = 0
    # C[0,1] (i=0, k=1) -> index 1*2+0 = 2? No.
    # Let's map Strassen C to Task W.
    # Strassen C[0,0] corresponds to A row 0, B col 0.
    # In Task: A row i, B col k.
    # Strassen A row 0 -> Task A row 0.
    # Strassen B col 0 -> Task B col 0?
    # In M_{2,2,2}, B is 2x2. Col 0 is index 0,1?
    # B[j, k].
    # Task V index: j*p + k.
    # For 2x2 block, j in {0,1}, k in {0,1}.
    # B[0,0] -> 0*2+0 = 0.
    # B[0,1] -> 0*2+1 = 1.
    # B[1,0] -> 1*2+0 = 2.
    # B[1,1] -> 1*2+1 = 3.
    # So e=f=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
    # My previous mapping: e,f,g,h -> 0,1,2,3.
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
    
    # C[0,0] (Task i=0, k=0) -> W index 0*2+0 = 0.
    # C[0,1] (Task i=0, k=1) -> W index 1*2+0 = 2.
    # C[1,0] (Task i=1, k=0) -> W index 0*2+1 = 1.
    # C[1,1] (Task i=1, k=1) -> W index 1*2+1 = 3.
    
    # So W indices for Strassen C:
    # C11 -> 0
    # C12 -> 2
    # C21 -> 1
    # C22 -> 3
    
    # Strassen Coefficients (P1..P7) for C11, C12, C21, C22
    # P1: 1, 0, 0, 1
    # P2: 0, 0, 1, -1
    # P3: 0, 1, 0, 1
    # P4: 1, 0, 1, 0
    # P5: -1, 1, 0, 0
    # P6: 0, 0, 0, 1
    # P7: 1, 0, 0, 0
    
    # Wait, let's re-verify the C mapping.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # W_coeff[product_idx][C_idx]
    # C_idx 0: C11
    # C_idx 1: C21 (index 1)
    # C_idx 2: C12 (index 2)
    # C_idx 3: C22 (index 3)
    
    W_coeffs = [
        [1, 0, 0, 1], # P1 -> C11, C21, C12, C22
        [0, 1, 0, -1], # P2
        [0, 0, 1, 1], # P3
        [1, 1, 0, 0], # P4
        [-1, 0, 1, 0], # P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # U_coeffs for A (a,b,c,d)
    U_coeffs = [
        [1, 0, 0, 1], # P1 (a,d)
        [0, 0, 1, 1], # P2 (c,d)
        [1, 0, 0, 0], # P3 (a)
        [0, 0, 0, 1], # P4 (d)
        [1, 1, 0, 0], # P5 (a,b)
        [-1, 0, 1, 0], # P6 (c,a) -> c-a
        [0, 1, 0, -1]  # P7 (b,d) -> b-d
    ]
    
    # V_coeffs for B (e,f,g,h)
    V_coeffs = [
        [1, 0, 0, 1], # P1 (e,h)
        [1, 0, 0, 0], # P2 (e)
        [0, 1, 0, -1], # P3 (f,h)
        [0, 0, 1, -1], # P4 (g,e)
        [0, 0, 0, 1], # P5 (h)
        [1, 1, 0, 0], # P6 (e,f)
        [0, 0, 1, 1]  # P7 (g,h)
    ]
    
    # We have 6 blocks of 2x2x2
    # 3 blocks for A_L B_T
    # 3 blocks for A_R B_B
    
    # Block 0: A_L, B_T Block 0
    # Block 1: A_L, B_T Block 1
    # Block 2: A_L, B_T Block 2
    # Block 3: A_R, B_B Block 0
    # Block 4: A_R, B_B Block 1
    # Block 5: A_R, B_B Block 2
    
    for block_idx in range(6):
        is_right = block_idx >= 3
        sub_block = block_idx % 3
        
        # Offset in U, V, W
        u_offset = 4 if is_right else 0
        v_offset = 12 + sub_block * 4 if is_right else sub_block * 4
        
        # W offset is always 0, but we add to the same W columns?
        # No, we have distinct columns in U, V, W for each of the 42 terms.
        # The "columns" in the decomposition matrices are the rank dimensions.
        # We are filling the matrices column by column for each product.
        
        for prod_idx in range(7):
            col = block_idx * 7 + prod_idx
            
            # Fill U
            for a_idx in range(4):
                U[u_offset + a_idx][col] = U_coeffs[prod_idx][a_idx]
                
            # Fill V
            for b_idx in range(4):
                V[v_offset + b_idx][col] = V_coeffs[prod_idx][b_idx]
                
            # Fill W
            # C indices: 0, 2, 1, 3 corresponding to C11, C12, C21, C22
            # W index mapping:
            # C11 -> 0
            # C12 -> 2
            # C21 -> 1
            # C22 -> 3
            w_map = [0, 2, 1, 3]
            for c_idx in range(4):
                W[w_map[c_idx]][col] = W_coeffs[prod_idx][c_idx]
                
    return {"U": U, "V": V, "W": W}
