import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Rank = 6 * 7 + 4 = 46
    rank = 46
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen blocks for pairs of columns
    # 6 pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for t in range(6):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # B indices:
        # B[0, k0] -> index 0*13 + k0
        # B[1, k0] -> index 1*13 + k0
        # B[0, k1] -> index 0*13 + k1
        # B[1, k1] -> index 1*13 + k1
        
        # A indices:
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[1,0] -> 2
        # A[1,1] -> 3
        
        # C indices:
        # C[0, k0] -> k0 * 2 + 0
        # C[0, k1] -> k1 * 2 + 0
        # C[1, k0] -> k0 * 2 + 1
        # C[1, k1] -> k1 * 2 + 1
        
        # M1 = (A00 + A11) * (B0_k0 + B1_k1)
        U[0, col] = 1; U[3, col] = 1
        V[0*13 + k0, col] = 1; V[1*13 + k1, col] = 1
        # C00 += M1
        W[k0 * 2 + 0, col] += 1
        # C11 += M1
        W[k1 * 2 + 1, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B0_k0
        U[2, col] = 1; U[3, col] = 1
        V[0*13 + k0, col] = 1
        # C10 += M2
        W[k0 * 2 + 1, col] += 1
        # C11 -= M2
        W[k1 * 2 + 1, col] -= 1
        col += 1
        
        # M3 = A00 * (B1_k0 - B1_k1)
        U[0, col] = 1
        V[1*13 + k0, col] = 1; V[1*13 + k1, col] = -1
        # C01 += M3
        W[k1 * 2 + 0, col] += 1
        # C11 += M3
        W[k1 * 2 + 1, col] += 1
        col += 1
        
        # M4 = A11 * (B0_k1 - B0_k0)
        U[3, col] = 1
        V[0*13 + k1, col] = 1; V[0*13 + k0, col] = -1
        # C00 += M4
        W[k0 * 2 + 0, col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B1_k1
        U[0, col] = 1; U[1, col] = 1
        V[1*13 + k1, col] = 1
        # C00 -= M5
        W[k0 * 2 + 0, col] -= 1
        # C01 += M5
        W[k1 * 2 + 0, col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B0_k0 + B0_k1)
        U[2, col] = 1; U[0, col] = -1
        V[0*13 + k0, col] = 1; V[0*13 + k1, col] = 1
        # C11 += M6
        W[k1 * 2 + 1, col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B1_k0 + B1_k1)
        U[1, col] = 1; U[3, col] = -1
        V[1*13 + k0, col] = 1; V[1*13 + k1, col] = 1
        # C00 -= M7
        W[k0 * 2 + 0, col] -= 1
        # C10 += M7
        W[k0 * 2 + 1, col] += 1
        col += 1

    # Rank-4 block for column 12
    k0 = 12
    
    # N1 = A00 * B0_k0
    U[0, col] = 1
    V[0*13 + k0, col] = 1
    W[k0 * 2 + 0, col] += 1
    col += 1
    
    # N2 = A01 * B1_k0
    U[1, col] = 1
    V[1*13 + k0, col] = 1
    W[k0 * 2 + 0, col] += 1
    col += 1
    
    # N3 = A10 * B0_k0
    U[2, col] = 1
    V[0*13 + k0, col] = 1
    W[k0 * 2 + 1, col] += 1
    col += 1
    
    # N4 = A11 * B1_k0
    U[3, col] = 1
    V[1*13 + k0, col] = 1
    W[k0 * 2 + 1, col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
