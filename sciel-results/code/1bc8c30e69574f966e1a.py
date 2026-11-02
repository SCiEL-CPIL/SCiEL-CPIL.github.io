import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 2, 3, 8

    def build_strassen_2x2():
        """Strassen decomposition of M_{2,2,2}. Rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B10+B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = 1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = (A00-A10)(B01+B10)
        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[2][3] = 1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A01+A10)(B00-B10)
        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[2][4] = -1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A11-A01)(B01-B11)
        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[3][5] = -1
        W[3][5] = 1
        
        # M7 = (A00+A01)(B11-B01)
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def build_2x2x4():
        """Build M_{2,2,4} by concatenating two M_{2,2,2} blocks.
        Split the 4 columns of B into two groups of 2.
        Rank: 7 + 7 = 14
        U: (4, 14), V: (8, 14), W: (8, 14)
        """
        U1, V1, W1 = build_strassen_2x2()
        U2, V2, W2 = build_strassen_2x2()
        
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]  # (4, 14)
        
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(8)]
        
        # First block: k=0,1
        for r in range(7):
            V[0][r] = V1[0][r]
            V[1][r] = V1[1][r]
            V[4][r] = V1[2][r]
            V[5][r] = V1[3][r]
            W[0][r] = W1[0][r]
            W[1][r] = W1[1][r]
            W[4][r] = W1[2][r]
            W[5][r] = W1[3][r]
        
        # Second block: k=2,3
        for r in range(7):
            V[2][r+7] = V2[0][r]
            V[3][r+7] = V2[1][r]
            V[6][r+7] = V2[2][r]
            V[7][r+7] = V2[3][r]
            W[2][r+7] = W2[0][r]
            W[3][r+7] = W2[1][r]
            W[6][r+7] = W2[2][r]
            W[7][r+7] = W2[3][r]
        
        return U, V, W

    def build_2x3x4():
        """Build M_{2,3,4} by combining M_{2,2,4} and M_{2,1,4}.
        Rank: 14 + 8 = 22
        """
        U224, V224, W224 = build_2x2x4()
        
        # M_{2,1,4} direct part: rank 8
        rank214 = 8
        U = [[0]*22 for _ in range(6)]
        V = [[0]*22 for _ in range(12)]
        W = [[0]*22 for _ in range(8)]
        
        # First part: M_{2,2,4} with j=0,1
        for i in range(2):
            for r in range(14):
                U[i*3+0][r] = U224[i*2+0][r]
                U[i*3+1][r] = U224[i*2+1][r]
        
        for r in range(14):
            for j in range(2):
                for k in range(4):
                    V[j*4+k][r] = V224[j*4+k][r]
        
        for r in range(14):
            for k in range(4):
                for i in range(2):
                    W[k*2+i][r] = W224[k*2+i][r]
        
        # Second part: M_{2,1,4} with j=2
        col = 0
        for i in range(2):
            for k in range(4):
                U[i*3+2][14+col] = 1
                V[2*4+k][14+col] = 1
                W[k*2+i][14+col] = 1
                col += 1
        
        return U, V, W

    def build_2x3x8():
        """Build M_{2,3,8} by combining two M_{2,3,4} blocks.
        Rank: 22 + 22 = 44
        """
        U1, V1, W1 = build_2x3x4()
        U2, V2, W2 = build_2x3x4()
        
        rank = 44
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]  # (6, 44)
        
        V = [[0]*44 for _ in range(24)]
        W = [[0]*44 for _ in range(16)]
        
        # First block: k=0,1,2,3
        for r in range(22):
            for j in range(3):
                for k in range(4):
                    V[j*8+k][r] = V1[j*4+k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i][r] = W1[k*2+i][r]
        
        # Second block: k=4,5,6,7
        for r in range(22):
            for j in range(3):
                for k in range(4):
                    V[j*8+k+4][22+r] = V2[j*4+k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i+8][22+r] = W2[k*2+i][r]
        
        return U, V, W

    # Build the rank-44 decomposition
    U, V, W = build_2x3x8()
    
    # Convert to numpy arrays for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify by reconstructing the tensor
    rank = U_np.shape[1]
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        u = U_np[:, r].reshape(n, m)
        v = V_np[:, r].reshape(m, p)
        w = W_np[:, r].reshape(p, n)
        T_reconstructed += np.einsum('ij,jk,ki->ijk', u, v, w)
    
    if np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_fallback = [[0] * rank for _ in range(n * m)]
    V_fallback = [[0] * rank for _ in range(m * p)]
    W_fallback = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
