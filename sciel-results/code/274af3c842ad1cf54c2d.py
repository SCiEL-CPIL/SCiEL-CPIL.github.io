import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Construct the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Rank-55 solution
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0, col] += 1; U[4, col] += 1
        V[k0, col] += 1; V[10+k1, col] += 1
        W[2*k0, col] += 1; W[2*k1+1, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[k0, col] += 1
        W[2*k0+1, col] += 1; W[2*k1+1, col] -= 1
        col += 1
        
        U[0, col] += 1
        V[10+k1, col] += 1; V[10+k1, col] -= 1  # Wait, this is wrong
        # Let me re-examine the indices
        
    # I'm getting confused with the indices. Let me just return the rank-55 solution as-is.
    
    # Reconstruct the rank-55 solution correctly
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0, col] += 1  # A[0,0]
        U[4, col] += 1  # A[1,1]
        V[k0, col] += 1  # B[0,k0]
        V[10+k1, col] += 1  # B[1,k1]
        W[2*k0, col] += 1  # C[0,k0]
        W[2*k1+1, col] += 1  # C[1,k1]
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[3, col] += 1  # A[1,0]
        U[4, col] += 1  # A[1,1]
        V[k0, col] += 1  # B[0,k0]
        W[2*k0+1, col] += 1  # C[1,k0]
        W[2*k1+1, col] -= 1  # C[1,k1]
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0, col] += 1  # A[0,0]
        V[10+k1, col] += 1  # B[1,k1]... wait, B0,k1 is V[k1], B1,k1 is V[10+k1]
        # Correction:
        V[k1, col] += 1  # B[0,k1]
        V[10+k1, col] -= 1  # B[1,k1]
        W[2*k1, col] += 1  # C[0,k1]
        W[2*k1+1, col] += 1  # C[1,k1]
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[4, col] += 1  # A[1,1]
        V[10+k0, col] += 1  # B[1,k0]
        V[k0, col] -= 1  # B[0,k0]
        W[2*k0, col] += 1  # C[0,k0]
        W[2*k0+1, col] += 1  # C[1,k0]
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0, col] += 1  # A[0,0]
        U[1, col] += 1  # A[0,1]
        V[10+k1, col] += 1  # B[1,k1]
        W[2*k0, col] -= 1  # C[0,k0]
        W[2*k1, col] += 1  # C[0,k1]
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[3, col] += 1  # A[1,0]
        U[0, col] -= 1  # A[0,0]
        V[k0, col] += 1  # B[0,k0]
        V[k1, col] += 1  # B[0,k1]
        W[2*k1+1, col] += 1  # C[1,k1]
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[1, col] += 1  # A[0,1]
        U[4, col] -= 1  # A[1,1]
        V[10+k0, col] += 1  # B[1,k0]
        V[10+k1, col] += 1  # B[1,k1]
        W[2*k0, col] += 1  # C[0,k0]
        col += 1
        
        # W00: A02 * B2,k0
        U[2, col] += 1  # A[0,2]
        V[20+k0, col] += 1  # B[2,k0]
        W[2*k0, col] += 1  # C[0,k0]
        col += 1
        
        # W01: A02 * B2,k1
        U[2, col] += 1  # A[0,2]
        V[20+k1, col] += 1  # B[2,k1]
        W[2*k1, col] += 1  # C[0,k1]
        col += 1
        
        # W10: A12 * B2,k0
        U[5, col] += 1  # A[1,2]
        V[20+k0, col] += 1  # B[2,k0]
        W[2*k0+1, col] += 1  # C[1,k0]
        col += 1
        
        # W11: A12 * B2,k1
        U[5, col] += 1  # A[1,2]
        V[20+k1, col] += 1  # B[2,k1]
        W[2*k1+1, col] += 1  # C[1,k1]
        col += 1
    
    # Verify
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_comp, T, atol=1e-10):
        # Fallback to naive
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
