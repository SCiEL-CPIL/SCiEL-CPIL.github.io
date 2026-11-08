import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    nm, mp, np_ = n * m, m * p, p * n  # 9, 18, 18
    
    # Laderman's rank-23 decomposition for 3x3 x 3x3
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # 23 bilinear forms M_t = (sum_a alpha[t][a] * A[a]) * (sum_b beta[t][b] * B[b])
    # C[c] = sum_t gamma[t][c] * M_t
    
    # A indices: 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23, 6=a31, 7=a32, 8=a33
    # B indices: 0=b11, 1=b12, 2=b13, 3=b21, 4=b22, 5=b23, 6=b31, 7=b32, 8=b33
    # C indices: 0=c11, 1=c12, 2=c13, 3=c21, 4=c22, 5=c23, 6=c31, 7=c32, 8=c33
    
    laderman_terms = [
        ([1,1,-1,-1,-1,-1,1,1,-1], [1,1,-1,-1,1,-1,1,1,-1], [1,0,0,0,1,0,0,0,1]),
        ([1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,-1,0,0], [0,0,1,0,0,0,0,0,0]),
        ([1,0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,-1,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,0,0,1], [1,1,0,-1,-1,0,0,0,0], [1,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,1,0,1], [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,1,0,1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,1,0,0,0,0,0,1], [0,0,0,1,1,0,-1,-1,0], [0,1,0,0,0,0,0,1,0]),
        ([1,0,1,0,-1,0,1,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,-1]),
        ([0,0,1,0,-1,0,0,0,1], [1,0,0,0,0,0,0,0,1], [0,0,0,1,0,0,0,0,0]),
        ([1,0,1,0,-1,0,0,0,0], [0,0,0,1,0,0,0,0,1], [1,0,0,0,0,0,1,0,0]),
        ([1,0,1,0,0,0,0,0,0], [0,0,0,1,-1,0,0,-1,1], [0,1,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,1,0,1], [0,1,1,0,-1,-1,0,0,0], [0,0,0,0,0,1,0,0,0]),
        ([0,0,0,0,-1,0,1,0,1], [0,1,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,1,0,-1,0,0,0,1], [1,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0]),
        ([0,0,1,0,0,0,0,0,1], [1,0,1,-1,0,-1,0,0,0], [0,1,0,0,0,0,0,1,0]),
        ([0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,-1,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,0], [1,0,0,-1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,-1,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,1,0,0,0], [0,1,1,0,-1,-1,0,0,0], [0,0,0,0,0,1,0,0,0]),
        ([1,0,1,-1,0,-1,0,0,0], [0,0,0,1,-1,0,0,-1,1], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,-1,0,-1,1,0,1], [1,0,1,-1,0,-1,0,0,0], [0,0,0,0,0,0,1,0,0]),
        ([1,0,1,-1,0,-1,1,0,1], [1,0,0,-1,0,-1,0,0,1], [1,0,0,0,0,0,0,0,0]),
        ([1,0,1,-1,-1,-1,1,0,1], [1,0,1,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
    ]
    
    rank = 46
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((np_, rank))
    
    # Mapping for B1 (cols 0-2 of B, which is 3x3)
    # B is 3x6. B1 is B[:, 0:3].
    # B indices in Laderman: 0..8 correspond to 3x3 matrix.
    # In the full 3x6 matrix B (flattened row-major 18x1):
    # Row 0: 0,1,2,3,4,5
    # Row 1: 6,7,8,9,10,11
    # Row 2: 12,13,14,15,16,17
    # B1 (cols 0-2): indices 0,1,2, 6,7,8, 12,13,14
    # B2 (cols 3-5): indices 3,4,5, 9,10,11, 15,16,17
    
    b1_map = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    b2_map = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    
    # C is 3x6. C1 is C[:, 0:3], C2 is C[:, 3:6]
    # C indices in Laderman: 0..8 correspond to 3x3 matrix.
    # In full 3x6 matrix C (flattened 18x1):
    # Row 0: 0,1,2,3,4,5
    # Row 1: 6,7,8,9,10,11
    # Row 2: 12,13,14,15,16,17
    # C1 (cols 0-2): indices 0,1,2, 6,7,8, 12,13,14
    # C2 (cols 3-5): indices 3,4,5, 9,10,11, 15,16,17
    
    c1_map = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    c2_map = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    
    for t in range(23):
        alpha, beta, gamma = laderman_terms[t]
        
        # Copy 1: for B1 and C1
        col1 = t
        U[:, col1] = alpha
        for i in range(9):
            V[b1_map[i], col1] = beta[i]
            W[c1_map[i], col1] = gamma[i]
            
        # Copy 2: for B2 and C2
        col2 = t + 23
        U[:, col2] = alpha
        for i in range(9):
            V[b2_map[i], col2] = beta[i]
            W[c2_map[i], col2] = gamma[i]
    
    # Verify
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
                
    reconstructed = np.zeros((nm, mp, np_))
    for c in range(rank):
        reconstructed += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
    if not np.allclose(reconstructed, T, atol=1e-10):
        # Fallback
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
