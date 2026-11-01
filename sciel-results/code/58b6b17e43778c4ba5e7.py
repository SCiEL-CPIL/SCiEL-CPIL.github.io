import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank = 39
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen blocks for k = 0, 2, 4, 6, 8
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices for block k:
    # b0 = 0*11 + k = k
    # b1 = 1*11 + k = 11+k
    # b2 = 0*11 + (k+1) = k+1
    # b3 = 1*11 + (k+1) = 11+k+1
    # C indices for block k:
    # c0 = k*2 + 0 = 2k
    # c1 = (k+1)*2 + 0 = 2k+2
    # c2 = k*2 + 1 = 2k+1
    # c3 = (k+1)*2 + 1 = 2k+3
    
    strassen_blocks = [0, 2, 4, 6, 8]
    
    for k in strassen_blocks:
        # B indices
        b0 = k
        b1 = 11 + k
        b2 = k + 1
        b3 = 11 + k + 1
        
        # C indices
        c0 = 2 * k
        c1 = 2 * (k + 1)
        c2 = 2 * k + 1
        c3 = 2 * (k + 1) + 1
        
        # A indices
        a0 = 0
        a1 = 1
        a2 = 2
        a3 = 3
        
        # 7 Strassen Multiplications
        # M1 = (a0+a1)*(b1+b3)
        U[a0, col] += 1; U[a1, col] += 1
        V[b1, col] += 1; V[b3, col] += 1
        W[c0, col] += 1; W[c1, col] += 1; W[c2, col] += 1; W[c3, col] += 1 
        # Wait, W coefficients must match the reconstruction.
        # C00 = M1 + M4 - M6 + M7
        # C01 = M3 + M6
        # C10 = M5 + M2
        # C11 = M1 - M2 + M3 - M5
        
        # Let's assign W carefully.
        # M1 contributes to C00 (+1), C11 (+1)
        # M2 contributes to C10 (+1), C11 (-1)
        # M3 contributes to C01 (+1), C11 (+1)
        # M4 contributes to C00 (+1)
        # M5 contributes to C10 (+1), C11 (-1)
        # M6 contributes to C00 (-1), C01 (+1)
        # M7 contributes to C00 (+1)
        
        # M1
        U[a0, col] = 1; U[a1, col] = 1
        V[b1, col] = 1; V[b3, col] = 1
        W[c0, col] = 1; W[c1, col] = 0; W[c2, col] = 0; W[c3, col] = 1
        col += 1
        
        # M2
        U[a0, col] = 1; U[a1, col] = -1
        V[b1, col] = 1; V[b3, col] = 1
        W[c0, col] = 0; W[c1, col] = 0; W[c2, col] = 1; W[c3, col] = -1
        col += 1
        
        # M3
        U[a2, col] = 1; U[a3, col] = 1
        V[b0, col] = 1; V[b2, col] = 1
        W[c0, col] = 0; W[c1, col] = 1; W[c2, col] = 0; W[c3, col] = 1
        col += 1
        
        # M4
        U[a2, col] = 1; U[a3, col] = 1
        V[b0, col] = 1; V[b2, col] = -1
        W[c0, col] = 1; W[c1, col] = 0; W[c2, col] = 0; W[c3, col] = 0
        col += 1
        
        # M5
        U[a0, col] = 1; U[a2, col] = 1
        V[b0, col] = 1
        W[c0, col] = 0; W[c1, col] = 0; W[c2, col] = 1; W[c3, col] = -1
        col += 1
        
        # M6
        U[a1, col] = 1; U[a3, col] = 1
        V[b3, col] = 1
        W[c0, col] = -1; W[c1, col] = 1; W[c2, col] = 0; W[c3, col] = 0
        col += 1
        
        # M7
        U[a0, col] = 1; U[a1, col] = 1
        V[b1, col] = 1
        W[c0, col] = 1; W[c1, col] = 0; W[c2, col] = 0; W[c3, col] = 0
        col += 1

    # Schoolbook block for k = 10
    k = 10
    b0 = k
    b1 = 11 + k
    c0 = 2 * k
    c1 = 2 * k + 1
    
    # M1 = a0 * b0
    U[a0, col] = 1
    V[b0, col] = 1
    W[c0, col] = 1
    col += 1
    
    # M2 = a1 * b1
    U[a1, col] = 1
    V[b1, col] = 1
    W[c0, col] = 1
    col += 1
    
    # M3 = a2 * b0
    U[a2, col] = 1
    V[b0, col] = 1
    W[c1, col] = 1
    col += 1
    
    # M4 = a3 * b1
    U[a3, col] = 1
    V[b1, col] = 1
    W[c1, col] = 1
    col += 1
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
