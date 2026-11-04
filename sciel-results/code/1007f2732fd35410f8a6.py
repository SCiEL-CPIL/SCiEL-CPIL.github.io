import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen factors for 2x2x2
    # U, V, W are 2x2 matrices
    S_U = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    S_V = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    S_W = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # Standard factors for 2x1x2
    # A is 2x1 (j=0 local), B is 1x2 (j=0 local), C is 2x2
    # T1: A00 B00 C00
    # T2: A00 B01 C01
    # T3: A10 B00 C10
    # T4: A10 B01 C11
    Std_U = [
        np.array([[1], [0]]),
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [1]])
    ]
    Std_V = [
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 0]]),
        np.array([[0, 1]])
    ]
    Std_W = [
        np.array([[1, 0], [0, 0]]), # C00 -> k=0, i=0
        np.array([[0, 0], [1, 0]]), # C01 -> k=1, i=0
        np.array([[0, 1], [0, 0]]), # C10 -> k=0, i=1
        np.array([[0, 0], [0, 1]])  # C11 -> k=1, i=1
    ]
    
    col = 0
    # There are 3 blocks of size 2 in the p dimension (6 = 3 * 2)
    for b in range(3):
        # Strassen part (handles A[:, 0:2] * B[0:2, 2b:2b+2])
        for r in range(7):
            # Map U (2x2) to global U (2x2 part of 2x3)
            # Local i, j in 0,1. Global i, j in 0,1.
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            
            # Map V (2x2) to global V (2x2 part of 3x2 block)
            # Local j, k' in 0,1. Global j in 0,1. Global k = 2b + k'.
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            
            # Map W (2x2) to global W (2x2 part of 2x2 block)
            # Local k', i in 0,1. Global k = 2b + k'. Global i in 0,1.
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            
            col += 1
            
        # Standard part (handles A[:, 2] * B[2, 2b:2b+2])
        for r in range(4):
            # Map U (2x1) to global U (2x1 part at j=2)
            for i in range(2):
                # Local j is 0, Global j is 2
                U[i * m + 2, col] = Std_U[r][i, 0]
            
            # Map V (1x2) to global V (1x2 part at j=2)
            for kp in range(2):
                k_global = 2 * b + kp
                # Local j is 0, Global j is 2
                V[2 * p + k_global, col] = Std_V[r][0, kp]
            
            # Map W (2x2) to global W
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp, i]
            
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
