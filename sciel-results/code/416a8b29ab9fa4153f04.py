import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 multiplication
    beta = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    gamma = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    delta = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # Reshape to 2x2 blocks for easier indexing
    beta_2d = beta.reshape(7, 2, 2)
    gamma_2d = gamma.reshape(7, 2, 2)
    delta_2d = delta.reshape(7, 2, 2)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U[i*4+j, col] where i = I*2+ip, j = J*2+jp
            # U = beta_s ⊗ beta_t (Kronecker product pattern)
            for I in range(2):
                for ip in range(2):
                    for J in range(2):
                        for jp in range(2):
                            row_u = (I * 2 + ip) * 4 + (J * 2 + jp)
                            U[row_u, col] += beta_2d[s, I, J] * beta_2d[t, ip, jp]
            
            # V[j*4+k, col] where j = J*2+jp, k = K*2+kp
            for J in range(2):
                for jp in range(2):
                    for K in range(2):
                        for kp in range(2):
                            row_v = (J * 2 + jp) * 4 + (K * 2 + kp)
                            V[row_v, col] += gamma_2d[s, J, K] * gamma_2d[t, jp, kp]
            
            # W[k*4+i, col] where k = K*2+kp, i = I*2+ip
            for K in range(2):
                for kp in range(2):
                    for I in range(2):
                        for ip in range(2):
                            row_w = (K * 2 + kp) * 4 + (I * 2 + ip)
                            W[row_w, col] += delta_2d[s, I, K] * delta_2d[t, ip, kp]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
