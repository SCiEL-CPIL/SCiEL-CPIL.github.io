import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # A[i,k] * B[k,j] -> C[i,j]
                # U index: i*m + k
                # V index: k*p + j
                # W index: j*n + i (Trace convention: W[k*n+i, r] in prompt? 
                # Prompt says: "output factor follows the official trace convention W[k*n+i,r]".
                # Usually W corresponds to C indices. C is n x p.
                # Let's check the indices in the failed snippet:
                # U[i*m+j, :] -> i is row of A, j is col of A (which is row of B). So U index i*m+j.
                # V[j*p+k, :] -> j is row of B, k is col of B (which is col of C). So V index j*p+k.
                # W[k*n+i, :] -> k is col of C, i is row of C. So W index k*n+i.
                # This matches the standard mapping:
                # A[i,j] * B[j,k] = C[i,k]
                # U[i, j] (flattened i*m+j)
                # V[j, k] (flattened j*p+k)
                # W[k, i] (flattened k*n+i) -> Note: W is p*n. Index k*n+i means row k, col i.
                # This is consistent with C[i,k] being formed by dotting U_row_i, V_row_j, W_row_k?
                # The contraction is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This equals C[i,k].
                
                U[i*m+k, idx] = 1.0
                V[k*p+j, idx] = 1.0
                W[j*n+i, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
