import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0 # 5 seconds safety margin before 120s
    
    n, m, p = 2, 2, 8
    rank = 28
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors for 2x2 x 2x2 -> 2x2
    # A is 2x2, indices: 00, 01, 10, 11
    # B is 2x2, indices: 00, 01, 10, 11
    # C is 2x2, indices: 00, 01, 10, 11
    
    # Strassen terms P1..P7
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # U vectors (coefficients for A)
    # Order: 00, 01, 10, 11
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    # V vectors (coefficients for B)
    # Order: 00, 01, 10, 11
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # W vectors (coefficients for C)
    # Order: 00, 01, 10, 11
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2 # 4 blocks of 2 columns
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # Fill U
            for i in range(4):
                U[i, col] = U_strassen[r][i]
                
            # Fill V
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
                
            # Fill W
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
                
            col += 1
            
    # Convert to lists of integers to satisfy "integer/half-integer" strictly
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    if time.time() > deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return {"U": U_list, "V": V_list, "W": W_list}
