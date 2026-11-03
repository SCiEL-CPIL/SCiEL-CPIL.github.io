import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2 (Strassen)
    # U, V, W shapes: (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # Strassen coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # c=0: M1
    U2[0, 0] = 1 # A11
    U2[3, 0] = 1 # A22
    V2[0, 0] = 1 # B11
    V2[3, 0] = 1 # B22
    W2[0, 0] = 1 # C11 += M1
    W2[3, 0] = 1 # C22 += M1
    
    # c=1: M2
    U2[2, 1] = 1 # A21
    U2[3, 1] = 1 # A22
    V2[0, 1] = 1 # B11
    W2[2, 1] = 1 # C21 += M2
    W2[3, 1] = -1 # C22 -= M2
    
    # c=2: M3
    U2[0, 2] = 1 # A11
    V2[1, 2] = 1 # B12
    V2[3, 2] = -1 # B22
    W2[1, 2] = 1 # C12 += M3
    W2[3, 2] = 1 # C22 += M3
    
    # c=3: M4
    U2[3, 3] = 1 # A22
    V2[2, 3] = 1 # B21
    V2[0, 3] = -1 # B11
    W2[0, 3] = 1 # C11 += M4
    W2[2, 3] = 1 # C21 += M4
    
    # c=4: M5
    U2[0, 4] = 1 # A11
    U2[1, 4] = 1 # A12
    V2[3, 4] = 1 # B22
    W2[0, 4] = -1 # C11 -= M5
    W2[1, 4] = 1 # C12 += M5
    
    # c=5: M6
    U2[2, 5] = 1 # A21
    U2[0, 5] = -1 # A11
    V2[0, 5] = 1 # B11
    V2[1, 5] = 1 # B12
    W2[3, 5] = 1 # C22 += M6
    
    # c=6: M7
    U2[1, 6] = 1 # A12
    U2[3, 6] = -1 # A22
    V2[2, 6] = 1 # B21
    V2[3, 6] = 1 # B22
    W2[0, 6] = 1 # C11 += M7
    
    # Now construct 4x4x4 decomposition of rank 49
    # U4, V4, W4 shapes: (16, 49), (16, 49), (16, 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # The 4x4 matrix is partitioned into 2x2 blocks.
    # A[i,j] for i,j in 0..3.
    # Block (bi, bj) contains A[2*bi + di, 2*bj + dj] for di,dj in 0..1.
    # U4 index: (2*bi + di)*4 + (2*bj + dj)
    # V4 index: (2*bj + dj)*4 + (2*bk + dk)
    # W4 index: (2*bk + dk)*4 + (2*bi + di)
    
    # For each Strassen term k (0..6), we have a 2x2x2 multiplication.
    # The 2x2x2 multiplication is applied to the blocks.
    # So for each k, we add 7 terms (from the 2x2x2 decomposition).
    
    for k in range(7):
        # k corresponds to one of the 7 Strassen products M_k
        # M_k is a 2x2 matrix product of blocks.
        # We use the U2, V2, W2 decomposition for this 2x2 product.
        for c2 in range(7):
            c = k * 7 + c2
            # U2[c2] gives coefficients for A blocks
            # V2[c2] gives coefficients for B blocks
            # W2[c2] gives coefficients for C blocks
            
            # U2 indices: 0=A11, 1=A12, 2=A21, 3=A22
            # V2 indices: 0=B11, 1=B12, 2=B21, 3=B22
            # W2 indices: 0=C11, 1=C12, 2=C21, 3=C22
            
            # We need to map block indices to element indices.
            # A block (bi, bj) -> elements (2*bi+di, 2*bj+dj)
            # U2[bi*2+bj, c2] is the coefficient for A block (bi, bj).
            
            for bi in range(2):
                for bj in range(2):
                    coeff_u = U2[bi*2 + bj, c2]
                    if coeff_u != 0:
                        for di in range(2):
                            for dj in range(2):
                                i = 2*bi + di
                                j = 2*bj + dj
                                U4[i*4 + j, c] = coeff_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_v = V2[bj*2 + bk, c2]
                    if coeff_v != 0:
                        for dj in range(2):
                            for dk in range(2):
                                j = 2*bj + dj
                                k_idx = 2*bk + dk
                                V4[j*4 + k_idx, c] = coeff_v
            
            for bk in range(2):
                for bi in range(2):
                    coeff_w = W2[bk*2 + bi, c2]
                    if coeff_w != 0:
                        for dk in range(2):
                            for di in range(2):
                                k_idx = 2*bk + dk
                                i = 2*bi + di
                                W4[k_idx*4 + i, c] = coeff_w
    
    # Now construct 4x4x3 schoolbook decomposition of rank 48
    # U5, V5, W5 shapes: (16, 48), (12, 48), (12, 48)
    # Wait, the second dimension is m=4, third is p=3.
    # U shape (n*m, rank) = (16, 48)
    # V shape (m*p, rank) = (12, 48)
    # W shape (p*n, rank) = (12, 48)
    
    U5 = np.zeros((16, 48), dtype=float)
    V5 = np.zeros((12, 48), dtype=float)
    W5 = np.zeros((12, 48), dtype=float)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U5[i*4 + j, col] = 1
                V5[j*3 + k, col] = 1
                W5[k*4 + i, col] = 1
                col += 1
    
    # Combine U4, U5 -> U
    # U4 is for p=4, U5 is for p=3.
    # U shape (16, 49+48) = (16, 97)
    U = np.hstack([U4, U5])
    
    # V4 is for p=4, V5 is for p=3.
    # V4 shape (16, 49), V5 shape (12, 48).
    # We need to pad V4 to have 12 rows? No.
    # The V matrix must have shape (m*p, rank) = (28, 97).
    # V4 corresponds to the first 4 columns of B (p=4 part).
    # V5 corresponds to the last 3 columns of B (p=3 part).
    # So V4 should be placed in rows 0..15 (j*4+k for k=0..3).
    # V5 should be placed in rows 16..27 (j*3+k for k=0..2, but shifted).
    # Wait, the index for V is j*p + k.
    # For p=7, j is 0..3, k is 0..6.
    # The first 4 columns (k=0..3) correspond to the 4x4 part.
    # The last 3 columns (k=4..6) correspond to the 4x3 part.
    
    # So for V4 (which is for the 4x4 part), the column index k in V4 is 0..3.
    # In the full V matrix, these correspond to k=0..3.
    # So V4[j*4+k, c] goes to V[j*7+k, c].
    
    V = np.zeros((28, 97), dtype=float)
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*7 + k, c] = V4[j*4 + k, c]
    
    # For V5 (which is for the 4x3 part), the column index k in V5 is 0..2.
    # In the full V matrix, these correspond to k=4..6.
    # So V5[j*3+k, c] goes to V[j*7 + (k+4), c+49].
    
    for j in range(4):
        for k in range(3):
            for c in range(48):
                V[j*7 + (k+4), c+49] = V5[j*3 + k, c]
    
    # W4 is for p=4, W5 is for p=3.
    # W shape (p*n, rank) = (28, 97).
    # W4[k*4+i, c] goes to W[k*4+i, c] for k=0..3.
    # W5[k*4+i, c] goes to W[(k+4)*4+i, c+49] for k=0..2.
    
    W = np.zeros((28, 97), dtype=float)
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4 + i, c] = W4[k*4 + i, c]
    
    for k in range(3):
        for i in range(4):
            for c in range(48):
                W[(k+4)*4 + i, c+49] = W5[k*4 + i, c]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
