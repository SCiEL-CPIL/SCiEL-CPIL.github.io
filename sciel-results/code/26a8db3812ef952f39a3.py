import numpy as np

def construct():
    m, n, p = 3, 5, 6
    # U shape: (n*m, rank) -> (15, R)
    # V shape: (m*p, rank) -> (18, R)
    # W shape: (p*n, rank) -> (30, R)
    
    rank = m * n * p  # 90
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(m):
        for j in range(p):
            for k in range(n):
                # Term for C[i,j] += A[i,k] * B[k,j]
                
                # U corresponds to A (size n*m = 15)
                # Index in A_flat for A[i,k] is i*n + k
                u_idx = i * n + k
                U[u_idx, r] = 1.0
                
                # W corresponds to B (size p*n = 30)
                # Index in B_flat for B[k,j] is k*p + j
                w_idx = k * p + j
                W[w_idx, r] = 1.0
                
                # V corresponds to C (size m*p = 18)
                # Index in C_flat for C[i,j] is i*p + j
                v_idx = i * p + j
                V[v_idx, r] = 1.0
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
