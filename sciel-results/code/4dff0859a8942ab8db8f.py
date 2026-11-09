"""Matrix multiplication 4x5x5: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Uses recursive block decomposition:
    r(4,5,5) <= 2 * r(2,5,5)  [split 4 rows into 2+2]
    r(2,5,5) <= r(2,5,4) + r(2,5,1) = r(2,5,4) + 10
    r(2,5,4) <= r(2,4,4) + r(2,1,4) = r(2,4,4) + 8
    r(2,4,4) <= 2 * r(2,4,2)  [split 4 cols into 2+2]
    r(2,4,2) <= 2 * r(2,2,2) = 2 * 7 = 14  [Strassen]
    So r(2,4,4) <= 28, r(2,5,4) <= 36, r(2,5,5) <= 46, r(4,5,5) <= 92
    """
    deadline = time.time() + 140.0
    
    # Strassen's 7-term decomposition for 2x2x2
    # C = A @ B where A, B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Strassen:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U is (2*2) x 7, V is (2*2) x 7, W is (2*2) x 7
        U indexed by [row, col] of A: row*2+col
        V indexed by [row, col] of B: row*2+col  
        W indexed by [row, col] of C: row*2+col
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (a11+a22)*(b11+b22)
        U[0][0] = 1; U[3][0] = 1  # a11, a22
        V[0][0] = 1; V[3][0] = 1  # b11, b22
        W[0][0] = 1; W[3][0] = 1  # C11, C22
        
        # m2 = (a21+a22)*b11
        U[2][1] = 1; U[3][1] = 1  # a21, a22
        V[0][1] = 1  # b11
        W[2][1] = 1; W[3][1] = -1  # C21, C22
        
        # m3 = a11*(b12-b22)
        U[0][2] = 1  # a11
        V[1][2] = 1; V[3][2] = -1  # b12, -b22
        W[1][2] = 1; W[3][2] = 1  # C12, C22
        
        # m4 = a22*(b21-b11)
        U[3][3] = 1  # a22
        V[2][3] = 1; V[0][3] = -1  # b21, -b11
        W[0][3] = 1; W[2][3] = 1  # C11, C21
        
        # m5 = (a11+a12)*b22
        U[0][4] = 1; U[1][4] = 1  # a11, a12
        V[3][4] = 1  # b22
        W[0][4] = -1; W[1][4] = 1  # -C11, C12
        
        # m6 = (a21-a11)*(b11+b12)
        U[2][5] = 1; U[0][5] = -1  # a21, -a11
        V[0][5] = 1; V[1][5] = 1  # b11, b12
        W[3][5] = 1  # C22
        
        # m7 = (a12-a22)*(b21+b22)
        U[1][6] = 1; U[3][6] = -1  # a12, -a22
        V[2][6] = 1; V[3][6] = 1  # b21, b22
        W[0][6] = 1  # C11
        
        return U, V, W
    
    def combine_decompositions(decs, n, m, p):
        """
        Combine a list of decompositions to form a larger one.
        Each dec is (U_i, V_i, W_i) for sub-problems.
        This is a placeholder - we build directly instead.
        """
        pass
    
    # Build the full 4x5x5 decomposition recursively
    # Strategy: 
    # 1. Split 4x5x5 into two 2x5x5 blocks (rows 0-1 and rows 2-3)
    # 2. Each 2x5x5: split into 2x5x4 + 2x5x1
    # 3. Each 2x5x4: split into 2x4x4 + 2x1x4  
    # 4. Each 2x4x4: split into two 2x4x2 blocks
    # 5. Each 2x4x2: split into two 2x2x2 blocks (Strassen)
    # 6. 2x5x1: rank 10 (each of 5 cols of B times each of 2 rows of A)
    # 7. 2x1x4: rank 8 (each of 4 cols of B times each of 2 rows of A)
    
    def build_2x2x2():
        """Strassen for 2x2x2. Returns U(4x7), V(4x7), W(4x7)"""
        return strassen_2x2x2()
    
    def build_2x4x2():
        """
        2x4x2: A is 2x4, B is 4x2, C is 2x2.
        Split B's 4 rows into two 2-row blocks: B = [[B1],[B2]]
        C = A1*B1 + A2*B2 where A1, A2 are 2x2 (columns of A)
        So r(2,4,2) <= r(2,2,2) + r(2,2,2) = 14
        U: (2*4) x 14 = 8 x 14
        V: (4*2) x 14 = 8 x 14
        W: (2*2) x 14 = 4 x 14
        """
        U1, V1, W1 = build_2x2x2()
        U2, V2, W2 = build_2x2x2()
        
        U = [[0]*14 for _ in range(8)]
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(4)]
        
        # First half: A cols 0-1 (indices 0-3 in U), B rows 0-1 (indices 0-3 in V)
        for i in range(4):
            for j in range(7):
                U[i][j] = U1[i][j]
                V[i][j] = V1[i][j]
                W[i][j] = W1[i][j]
        
        # Second half: A cols 2-3 (indices 4-7 in U), B rows 2-3 (indices 4-7 in V)
        for i in range(4):
            for j in range(7):
                U[i+4][j+7] = U2[i][j]
                V[i+4][j+7] = V2[i][j]
                W[i][j+7] = W2[i][j]
        
        return U, V, W
    
    def build_2x4x4():
        """
        2x4x4: A is 2x4, B is 4x4, C is 2x4.
        Split B's 4 cols into two 2-col blocks: B = [B1 | B2]
        C = [A*B1 | A*B2]
        r(2,4,4) <= r(2,4,2) + r(2,4,2) = 28
        U: (2*4) x 28 = 8 x 28
        V: (4*4) x 28 = 16 x 28
        W: (4*2) x 28 = 8 x 28
        """
        U1, V1, W1 = build_2x4x2()
        U2, V2, W2 = build_2x4x2()
        
        U = [[0]*28 for _ in range(8)]
        V = [[0]*28 for _ in range(16)]
        W = [[0]*28 for _ in range(8)]
        
        # First half: B cols 0-1 (V indices 0-7), C cols 0-1 (W indices 0-3)
        for i in range(8):
            for j in range(14):
                U[i][j] = U1[i][j]
                V[i][j] = V1[i][j]
                W[i][j] = W1[i][j]
        
        # Second half: B cols 2-3 (V indices 8-15), C cols 2-3 (W indices 4-7)
        for i in range(8):
            for j in range(14):
                U[i][j+14] = U2[i][j]
                V[i+8][j+14] = V2[i][j]
                W[i+4][j+14] = W2[i][j]
        
        return U, V, W
    
    def build_2x1x4():
        """
        2x1x4: A is 2x1, B is 1x4, C is 2x4.
        C[i,k] = A[i,0] * B[0,k]
        Rank = 2*4 = 8
        U: (2*1) x 8 = 2 x 8
        V: (1*4) x 8 = 4 x 8
        W: (4*2) x 8 = 8 x 8
        """
        rank = 8
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(8)]
        
        col = 0
        for i in range(2):
            for k in range(4):
                U[i][col] = 1
                V[k][col] = 1
                W[k*2+i][col] = 1
                col += 1
        
        return U, V, W
    
    def build_2x5x4():
        """
        2x5x4: A is 2x5, B is 5x4, C is 2x4.
        Split A's 5 cols into 4+1: A = [A1 | A2]
        C = A1*B1 + A2*B2 where B1 is 4x4, B2 is 1x4
        r(2,5,4) <= r(2,4,4) + r(2,1,4) = 28 + 8 = 36
        U: (2*5) x 36 = 10 x 36
        V: (5*4) x 36 = 20 x 36
        W: (4*2) x 36 = 8 x 36
        """
        U1, V1, W1 = build_2x4x4()
        U2, V2, W2 = build_2x1x4()
        
        U = [[0]*36 for _ in range(10)]
        V = [[0]*36 for _ in range(20)]
        W = [[0]*36 for _ in range(8)]
        
        # First part: A cols 0-3 (U indices 0-7), B rows 0-3 (V indices 0-15)
        for i in range(8):
            for j in range(28):
                U[i][j] = U1[i][j]
                V[i][j] = V1[i][j]
                W[i][j] = W1[i][j]
        
        # Second part: A col 4 (U index 8), B row 4 (V index 16)
        for i in range(2):
            for j in range(8):
                U[i+8][j+28] = U2[i][j]
                V[i+16][j+28] = V2[i][j]
                W[i][j+28] = W2[i][j]
        
        return U, V, W
    
    def build_2x5x1():
        """
        2x5x1: A is 2x5, B is 5x1, C is 2x1.
        C[i,0] = sum_j A[i,j] * B[j,0]
        Rank = 2*5 = 10
        U: (2*5) x 10 = 10 x 10
        V: (5*1) x 10 = 5 x 10
        W: (1*2) x 10 = 2 x 10
        """
        rank = 10
        U = [[0]*rank for _ in range(10)]
        V = [[0]*rank for _ in range(5)]
        W = [[0]*rank for _ in range(2)]
        
        col = 0
        for i in range(2):
            for j in range(5):
                U[i*5+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        
        return U, V, W
    
    def build_2x5x5():
        """
        2x5x5: A is 2x5, B is 5x5, C is 2x5.
        Split B's 5 cols into 4+1: B = [B1 | B2]
        C = [A*B1 | A*B2]
        r(2,5,5) <= r(2,5,4) + r(2,5,1) = 36 + 10 = 46
        U: (2*5) x 46 = 10 x 46
        V: (5*5) x 46 = 25 x 46
        W: (5*2) x 46 = 10 x 46
        """
        U1, V1, W1 = build_2x5x4()
        U2, V2, W2 = build_2x5x1()
        
        U = [[0]*46 for _ in range(10)]
        V = [[0]*46 for _ in range(25)]
        W = [[0]*46 for _ in range(10)]
        
        # First part: B cols 0-3 (V indices 0-19), C cols 0-3 (W indices 0-7)
        for i in range(10):
            for j in range(36):
                U[i][j] = U1[i][j]
                V[i][j] = V1[i][j]
                W[i][j] = W1[i][j]
        
        # Second part: B col 4 (V indices 20-24), C col 4 (W indices 8-9)
        for i in range(10):
            for j in range(10):
                U[i][j+36] = U2[i][j]
                V[i+20][j+36] = V2[i][j]
                W[i+8][j+36] = W2[i][j]
        
        return U, V, W
    
    def build_4x5x5():
        """
        4x5x5: A is 4x5, B is 5x5, C is 4x5.
        Split A's 4 rows into 2+2: A = [[A1],[A2]]
        C = [[A1*B],[A2*B]]
        r(4,5,5) <= 2 * r(2,5,5) = 2 * 46 = 92
        U: (4*5) x 92 = 20 x 92
        V: (5*5) x 92 = 25 x 92
        W: (5*4) x 92 = 20 x 92
        """
        U1, V1, W1 = build_2x5x5()
        U2, V2, W2 = build_2x5x5()
        
        U = [[0]*92 for _ in range(20)]
        V = [[0]*92 for _ in range(25)]
        W = [[0]*92 for _ in range(20)]
        
        # First block: A rows 0-1 (U indices 0-9), C rows 0-1 (W indices 0-9)
        for i in range(10):
            for j in range(46):
                U[i][j] = U1[i][j]
                V[i][j] = V1[i][j]
                W[i][j] = W1[i][j]
        
        # Second block: A rows 2-3 (U indices 10-19), C rows 2-3 (W indices 10-19)
        for i in range(10):
            for j in range(46):
                U[i+10][j+46] = U2[i][j]
                V[i][j+46] = V2[i][j]
                W[i+10][j+46] = W2[i][j]
        
        return U, V, W
    
    U, V, W = build_4x5x5()
    
    # Verify correctness with a random test
    n, m, p = 4, 5, 5
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # C = sum over rank: U[:,r] outer V[:,r] outer W[:,r]
    # C[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total += np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
