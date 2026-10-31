import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def _matmul_tensor(n, m, p):
    """Tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matmul tensor."""
    T = _matmul_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 via Strassen (2 levels of 7)
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 block multiplication
    a_coeffs = [
        [[1, 0], [0, 1]],  # M0
        [[0, 0], [1, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 0], [0, 1]],  # M3
        [[1, 1], [0, 0]],  # M4
        [[-1, 0], [1, 0]], # M5
        [[0, 1], [0, -1]], # M6
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],  # M0
        [[1, 0], [0, 0]],  # M1
        [[0, 1], [0, -1]], # M2
        [[-1, 0], [1, 0]], # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, 1]],  # M6
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],  # M0 -> C00, C11
        [[0, 0], [1, -1]], # M1 -> C10, C11
        [[0, 1], [0, 0]],  # M2 -> C01
        [[-1, 0], [0, 1]], # M3 -> C00, C10
        [[1, 1], [0, 0]],  # M4 -> C00, C01
        [[0, 0], [0, 1]],  # M5 -> C11
        [[0, 0], [0, 1]],  # M6 -> C11
    ]
    
    # Build 2x2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    for s in range(7):
        for I in range(2):
            for J in range(2):
                if a_coeffs[s][I][J] != 0:
                    U2[I*2+J, s] = a_coeffs[s][I][J]
        for J in range(2):
            for K in range(2):
                if b_coeffs[s][J][K] != 0:
                    V2[J*2+K, s] = b_coeffs[s][J][K]
        for I in range(2):
            for K in range(2):
                if c_coeffs[s][I][K] != 0:
                    W2[K*2+I, s] = c_coeffs[s][I][K]
    
    # Assemble 4x4x4
    for s in range(7):
        for t in range(7):
            col = 7*s + t
            for i in range(4):
                I = i // 2
                i_prime = i % 2
                for j in range(4):
                    J = j // 2
                    j_prime = j % 2
                    row_u = i*4 + j
                    val = a_coeffs[s][I][J] * U2[i_prime*2 + j_prime, t]
                    if val != 0:
                        U[row_u, col] = val
                        
            for j in range(4):
                J = j // 2
                j_prime = j % 2
                for k in range(4):
                    K = k // 2
                    k_prime = k % 2
                    row_v = j*4 + k
                    val = b_coeffs[s][J][K] * V2[j_prime*2 + k_prime, t]
                    if val != 0:
                        V[row_v, col] = val
                        
            for k in range(4):
                K = k // 2
                k_prime = k % 2
                for i in range(4):
                    I = i // 2
                    i_prime = i % 2
                    row_w = k*4 + i
                    val = c_coeffs[s][I][K] * W2[k_prime*2 + i_prime, t]
                    if val != 0:
                        W[row_w, col] = val
                        
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
