import time
import numpy as np

def construct():
    """
    Returns a rank-28 tensor decomposition for multiplying a 2x2 matrix by a 2x8 matrix.
    This is based on the Strassen algorithm applied to 4 independent 2x2x2 blocks.
    """
    n, m, p = 2, 2, 8
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    # U, V, W are lists of 7 vectors of length 4 (flattened 2x2 matrices)
    # Convention: Index in flattened matrix is i*n + j for A (2x2), so i*2+j.
    # For B (2x2), index is j*p + k? No, in the block, B is 2x2.
    # In the full problem, B is 2x8.
    
    # Standard Strassen decomposition for 2x2x2
    # T = sum_{r=0}^6 u_r (A) * v_r (B) * w_r (C)
    # A is 2x2, B is 2x2, C is 2x2.
    
    U_s = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1],
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]
    ]
    V_s = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]
    ]
    W_s = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0],
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]
    ]

    rank = 28
    num_blocks = 4 # 8 columns / 2 columns per block
    
    U = np.zeros((n * m, rank), dtype=float) # 4 x 28
    V = np.zeros((m * p, rank), dtype=float) # 16 x 28
    W = np.zeros((p * n, rank), dtype=float) # 16 x 28
    
    for t in range(num_blocks):
        base_col = t * 7
        for r in range(7):
            col = base_col + r
            # A part: same for all blocks
            for i in range(4):
                U[i, col] = U_s[r][i]
            
            # B part: block t corresponds to columns 2*t and 2*t+1 of B
            # B is 2x8. Index in V is j*8 + k.
            # Block t uses k=2*t and k=2*t+1.
            # V_s[r] is a vector of length 4 for a 2x2 matrix.
            # The 2x2 block B_t has entries B[j, 2*t + l] for j,l in {0,1}.
            # Index in V for B[j, k] is j*8 + k.
            # So for l=0 (k=2*t), indices are 0*8 + 2*t and 1*8 + 2*t.
            # For l=1 (k=2*t+1), indices are 0*8 + 2*t+1 and 1*8 + 2*t+1.
            # V_s[r] is ordered [b00, b01, b10, b11]?
            # Let's check the standard Strassen vector ordering.
            # Usually flattened row-major: [a00, a01, a10, a11].
            # So V_s[r][0] is b00, V_s[r][1] is b01, V_s[r][2] is b10, V_s[r][3] is b11.
            # In our full V matrix:
            # b00 is at index 0*8 + 2*t = 2*t
            # b01 is at index 0*8 + 2*t+1 = 2*t+1
            # b10 is at index 1*8 + 2*t = 8 + 2*t
            # b11 is at index 1*8 + 2*t+1 = 8 + 2*t+1
            
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_s[r][l]
            
            # W part: block t corresponds to columns 2*t and 2*t+1 of C
            # C is 2x8. Index in W is k*n + i = k*2 + i.
            # Block t uses k=2*t and k=2*t+1.
            # W_s[r] is a vector of length 4 for a 2x2 matrix C_t.
            # W_s[r] ordered [c00, c01, c10, c11].
            # c00 is at index (2*t)*2 + 0 = 4*t
            # c01 is at index (2*t)*2 + 1 = 4*t+1
            # c10 is at index (2*t+1)*2 + 0 = 4*t+2
            # c11 is at index (2*t+1)*2 + 1 = 4*t+3
            
            w_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_s[r][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
