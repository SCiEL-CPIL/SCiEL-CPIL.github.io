import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # I = i*m + j
                # J = j*p + k
                # K = k*n + i
                I = i * m + j
                J = j * p + k
                K = k * n + i
                
                U[I, col] = 1
                V[J, col] = 1
                W[K, col] = 1
                
                col += 1
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

if __name__ == "__main__":
    res = construct()
    n, m, p = 3, 4, 8
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                u_val = U[I]
                v_val = V[J]
                w_val = W[K]
                dot = np.dot(u_val, v_val * w_val)
                # Target is 1 if I,J,K form a valid MM term
                # Which they do by construction
                if abs(dot - 1.0) > max_err:
                    max_err = abs(dot - 1.0)
    print(f"Max Error: {max_err}")
