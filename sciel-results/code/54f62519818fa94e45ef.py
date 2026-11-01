import numpy as np

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients
    # t=0..6
    S_A = [
        np.array([[1, 0], [0, 1]]), # P1
        np.array([[0, 0], [1, 1]]), # P2
        np.array([[1, 0], [0, 0]]), # P3
        np.array([[0, 0], [0, 1]]), # P4
        np.array([[1, 0], [1, 0]]), # P5
        np.array([[-1, 0], [1, 0]]),# P6
        np.array([[0, 1], [0, -1]]) # P7
    ]
    
    S_B = [
        np.array([[1, 0], [0, 1]]), # P1
        np.array([[1, 0], [0, 0]]), # P2
        np.array([[0, 0], [1, -1]]),# P3
        np.array([[-1, 1], [0, 0]]),# P4
        np.array([[0, 0], [0, 1]]), # P5
        np.array([[1, 0], [1, 0]]), # P6
        np.array([[0, 1], [1, 1]])  # P7
    ]
    
    S_C = [
        np.array([[1, 0], [0, 1]]), # P1 -> C00, C11
        np.array([[0, 0], [1, -1]]),# P2 -> C10, C11
        np.array([[0, 1], [0, 1]]), # P3 -> C01, C11
        np.array([[1, 0], [1, 0]]), # P4 -> C00, C10
        np.array([[-1, 1], [0, 0]]),# P5 -> C00, C01
        np.array([[0, 0], [0, 1]]), # P6 -> C11
        np.array([[1, 0], [0, 0]])  # P7 -> C00
    ]
    
    for b in range(4):
        for t in range(7):
            r = b * 7 + t
            
            # Fill U
            # U index: i*m + j
            for i in range(n):
                for j in range(m):
                    U[i*m + j, r] = S_A[t][i, j]
                    
            # Fill V
            # V index: j*p + k
            # Block b uses columns 2b, 2b+1
            for j in range(m):
                for k_local in range(2):
                    k_global = 2*b + k_local
                    V[j*p + k_global, r] = S_B[t][j, k_local]
                    
            # Fill W
            # W index: k*n + i
            # Block b uses columns 2b, 2b+1
            for k_local in range(2):
                k_global = 2*b + k_local
                for i in range(n):
                    W[k_global*n + i, r] = S_C[t][i, k_local]
                    
    return {'U': U, 'V': V, 'W': W}
