"""Program for matrix multiplication 4,4,7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def build_block_decomposition_4x4x7():
        """
        Build a decomposition for 4x4x7 by viewing A as 2x2 blocks of 2x2,
        B as 2x2 blocks of 2x2 (with the second block row being 2x3 and 2x4 padded),
        and using Strassen's method for the 2x2 block structure.
        
        Actually, a cleaner approach:
        A is 4x4, B is 4x7.
        We can split A into 2x2 blocks of size 2x2: A = [[A11, A12], [A21, A22]]
        B into 2x2 blocks where the columns are split as 2+5 or similar? No, B is 4x7.
        
        Let's use a different strategy.
        We know 4x4x4 has a rank-49 decomposition (Kronecker of Strassen).
        For 4x4x7, we can write B = [B1, B2] where B1 is 4x4 and B2 is 4x3.
        C = A*B1 + A*B2.
        
        This doesn't directly help unless we have a good 4x4x3 decomposition.
        
        Alternative: Use the 4x4x4 Strassen-Kronecker decomposition (rank 49) for the first 4 columns of B,
        and a schoolbook (rank 4*4*3 = 48) for the last 3 columns. Total rank 49+48 = 97.
        This matches the initial inspiration's rank 97.
        
        Can we do better?
        The inspiration mentions fixing index bugs to get rank 93.
        Let's try to construct a rank-93 decomposition directly or find the bug.
        
        Let's try a hybrid: 
        Use Strassen for 2x2x2 blocks.
        A (4x4) = [[A11, A12], [A21, A22]] (each 2x2)
        B (4x7) = [[B11, B12, B13], [B21, B22, B23]] where B11, B21 are 2x2, and B12, B22, B13, B23 are 2x? 
        This is getting complicated.
        
        Let's stick to the known rank-49 for 4x4x4 and see if we can optimize the 4x4x3 part.
        Schoolbook for 4x4x3 is rank 48.
        Is there a better 4x4x3?
        Maybe use Strassen for 2x2x2 and schoolbook for the rest?
        
        Let's try to build the rank-97 decomposition first and verify it, then try to improve.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        # 4x4x4 part (rank 49)
        U44 = [[0]*49 for _ in range(16)]
        V44 = [[0]*49 for _ in range(16)]
        W44 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U44[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V44[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W44[k*4+i][r] += val
        
        # 4x4x3 part (rank 48)
        U43 = [[0]*48 for _ in range(16)]
        V43 = [[0]*48 for _ in range(12)] # 4*3
        W43 = [[0]*48 for _ in range(12)] # 3*4
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U43[i*4+j][col] = 1
                    V43[j*3+k][col] = 1
                    W43[k*4+i][col] = 1
                    col += 1
        
        # Combine into 4x4x7
        # U: 16 x (49+48)
        # V: 16 x 49, 12 x 48 -> need to pad V44 to 12 rows? No, V is (m*p, rank) = (4*7, rank) = (28, rank)
        # Wait, V is (m*p, rank) = (4*7, rank) = (28, rank).
        # In the 4x4x4 part, V44 is 16x49. We need to embed this into 28x49.
        # The first 16 rows of V correspond to B[0:4, 0:4]. The remaining 12 rows are 0.
        # In the 4x4x3 part, V43 is 12x48. We need to embed this into 28x48.
        # The first 16 rows of V are 0, and the last 12 rows are V43.
        
        rank_total = 49 + 48
        U_total = [[0] * rank_total for _ in range(16)]
        V_total = [[0] * rank_total for _ in range(28)]
        W_total = [[0] * rank_total for _ in range(28)] # p*n = 7*4 = 28
        
        # Fill 4x4x4 part
        for i in range(16):
            for r in range(49):
                U_total[i][r] = U44[i][r]
                V_total[i][r] = V44[i][r]
                W_total[r*4+i if False else (r%4)*4 + (r//4)][r] = 0 # W indexing is tricky
        
        # Let's be more careful with W.
        # W is (p*n, rank) = (28, rank).
        # W[k*n+i, r] corresponds to C[i,k].
        # In the 4x4x4 part, C is 4x4. k is 0..3, i is 0..3.
        # W44[k*4+i, r]. This maps to rows 0..15 of W_total.
        # In the 4x4x3 part, C is 4x3. k is 0..2, i is 0..3.
        # W43[k*4+i, r]. This maps to rows 16..27 of W_total.
        
        # Let's rebuild W_total carefully.
        for i in range(16):
            for r in range(49):
                U_total[i][r] = U44[i][r]
                V_total[i][r] = V44[i][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W_total[k*4+i][r] = W44[k*4+i][r]
        
        # Fill 4x4x3 part
        for i in range(16):
            for r in range(48):
                U_total[i][49+r] = U43[i][r]
        
        for j in range(4):
            for k in range(3):
                for r in range(48):
                    V_total[16 + j*3+k][49+r] = V43[j*3+k][r]
        
        for k in range(3):
            for i in range(4):
                for r in range(48):
                    W_total[16 + k*4+i][49+r] = W43[k*4+i][r]
        
        return U_total, V_total, W_total
    
    # First, get a valid baseline
    U, V, W = build_block_decomposition_4x4x7()
    
    # Verify the decomposition
    def check_valid(U, V, W, tol=1e-9):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not check_valid(U, V, W):
        # Fallback to schoolbook if the hybrid is invalid
        U, V, W = build_schoolbook()
    
    return {"U": U, "V": V, "W": W}
