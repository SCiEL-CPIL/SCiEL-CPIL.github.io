import numpy as np

def construct():
    n, m, p = 2, 2, 10
    num_blocks = 5
    rank = 35
    
    # Strassen factors for 2x2
    # U: 4x7
    U_base = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    # V: 4x7
    V_base = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ])
    
    # W: 4x7
    W_base = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Initialize full matrices
    U = np.zeros((4, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    for t in range(num_blocks):
        col_start = t * 7
        col_end = (t + 1) * 7
        
        # Fill U
        U[:, col_start:col_end] = U_base
        
        # Fill V
        # V indices: j*p + k
        # Block t uses k=2t, 2t+1
        # j=0: indices 2t, 2t+1
        # j=1: indices 10+2t, 10+2t+1
        V[2*t, col_start:col_end] = V_base[0]
        V[2*t+1, col_start:col_end] = V_base[1]
        V[10+2*t, col_start:col_end] = V_base[2]
        V[10+2*t+1, col_start:col_end] = V_base[3]
        
        # Fill W
        # W indices: k*n + i
        # Block t uses k=2t, 2t+1
        # k=2t: indices 4t, 4t+1
        # k=2t+1: indices 4t+2, 4t+3
        W[4*t, col_start:col_end] = W_base[0]
        W[4*t+1, col_start:col_end] = W_base[1]
        W[4*t+2, col_start:col_end] = W_base[2]
        W[4*t+3, col_start:col_end] = W_base[3]
        
    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
