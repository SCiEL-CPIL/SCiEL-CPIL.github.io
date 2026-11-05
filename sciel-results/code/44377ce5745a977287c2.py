"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a dictionary containing the factor matrices U, V, W for the
    tensor decomposition of the matrix multiplication tensor for
    (n x m) * (m x p) matrices.
    Uses Strassen's algorithm for 4x4x4 part and schoolbook for 4x4x1 part.
    Total rank: 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS
    
    # 1. Construct M_{2,2,2} Strassen decomposition
    # U, V, W shapes (4, 7)
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # P1: (A11+A22)(B11+B22) -> C11+C22
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # P2: (A21+A22)B11 -> C21
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[1, 1] = 1 # C10 (index 1)
    
    # P3: A11(B12-B22) -> C12+C22
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[2, 2] = 1; W22[3, 2] = 1 # C01 (idx 2), C11 (idx 3)
    
    # P4: A22(B21-B11) -> C11+C21
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[1, 3] = 1 # C00 (idx 0), C10 (idx 1)
    
    # P5: (A11+A12)B22 -> -C11+C12
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[2, 4] = 1
    
    # P6: (A21-A11)(B11+B12) -> C22
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1
    
    # P7: (A12-A22)(B21+B22) -> C11+C22
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1; W22[3, 6] = 1
    
    # 2. Construct M_{4,4,4} from M_{2,2,2}
    # Rank 49
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            u1 = U22[:, r1]
            u2 = U22[:, r2]
            v1 = V22[:, r1]
            v2 = V22[:, r2]
            w1 = W22[:, r1]
            w2 = W22[:, r2]
            
            # Construct Kronecker-like vectors with correct index mapping
            # Index 44: i*4+j = (2*i1+i0)*4 + (2*j1+j0) = 8*i1 + 4*i0 + 2*j1 + j0
            # u1 index: i1*2+j1, u2 index: i0*2+j0
            for i1 in range(2):
                for j1 in range(2):
                    idx1 = i1 * 2 + j1
                    val1 = u1[idx1]
                    if val1 == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            idx2 = i0 * 2 + j0
                            val2 = u2[idx2]
                            if val2 == 0: continue
                            idx44 = 8 * i1 + 4 * i0 + 2 * j1 + j0
                            U44[idx44, col] = val1 * val2
                            V44[idx44, col] = v1[idx1] * v2[idx2]
                            W44[idx44, col] = w1[idx1] * w2[idx2]
                            
    # 3. Construct M_{4,4,1} schoolbook
    # Rank 16
    U441 = np.zeros((16, 16))
    V441 = np.zeros((4, 16))
    W441 = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U441[i * 4 + j, r] = 1
            V441[j, r] = 1
            W441[i, r] = 1
            
    # 4. Combine into M_{4,4,5}
    # n=4, m=4, p=5
    # U shape (16, 65)
    # V shape (20, 65)
    # W shape (20, 65)
    
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Fill M_{4,4,4} part (cols 0-48)
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, 0:49] = V44[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W44[k * 4 + i, 0:49]
            
    # Fill M_{4,4,1} part (cols 49-64)
    U[:, 49:65] = U441
    for j in range(4):
        V[j * 5 + 4, 49:65] = V441[j, :]
    for i in range(4):
        W[4 * 4 + i, 49:65] = W441[i, :]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
