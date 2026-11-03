import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid using numpy for speed."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_reshaped = U_arr.reshape(n, m, r)
        V_reshaped = V_arr.reshape(m, p, r)
        W_reshaped = W_arr.reshape(p, n, r)
        
        # Compute the sum over r for all (i, j, k)
        # U[i,j,:] * V[j,k,:] * W[k,i,:]
        # We can do this by iterating over i, j, k or using einsum
        # Let's try einsum: sum over r
        # Result[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a bit tricky with standard einsum because indices are permuted.
        # Let's just do a loop over i, j, k but vectorize over r?
        # Or just do the triple loop, it's 4*4*7 = 112 iterations.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    # Vectorized sum over r
                    s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :]) * 1.0 # Wait, W is also involved.
                    # s = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Let A = U[i,j,:], B = V[j,k,:], C = W[k,i,:]
                    # s = sum_r A[r]*B[r]*C[r]
                    # This is element-wise product then sum.
                    prod = U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True

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

    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
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
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4

    def build_4x4x7_decomposition():
        """
        Build a rank-93 decomposition for 4x4x7.
        
        Strategy: Use Kronecker Strassen (rank 49) for the 4x4x4 part.
        For the 4x4x3 part (columns 4,5,6 of B), use schoolbook for each column.
        Schoolbook for 4x4x1 is rank 16.
        3 columns * 16 = 48.
        Total rank = 49 + 48 = 97.
        
        Wait, the previous program used 2x2x3 blocks of rank 11.
        4 blocks * 11 = 44.
        Total rank = 49 + 44 = 93.
        Let's stick with the 93 construction if it's valid and fast.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        # For the remaining 3 columns (indices 4,5,6 in B), we can use schoolbook for 4x4x3.
        # Rank of 4x4x3 schoolbook is 4*4*3 = 48.
        # Total rank = 49 + 48 = 97.
        
        # Alternatively, use the 2x2x3 block decomposition.
        # A 2x2x3 matrix multiplication (A 2x2, B 2x3) has rank 11 (7 for 2x2x2 + 4 for 2x2x1).
        # We have 4 such blocks (2x2 grid of A blocks).
        # Total rank for the 4x4x3 part = 4 * 11 = 44.
        # Total rank = 49 + 44 = 93.
        
        # Let's implement the 2x2x3 block decomposition.
        
        def build_2x2x3():
            """
            Build rank-11 decomposition for 2x2x3 matrix multiplication.
            A is 2x2, B is 2x3, C is 2x2.
            U: 4x11 (indices i*2+j), V: 6x11 (indices j*3+k), W: 6x11 (indices k*2+i)
            """
            U = [[0]*11 for _ in range(4)]
            V = [[0]*11 for _ in range(6)]
            W = [[0]*11 for _ in range(6)]
            
            # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
            U2, V2, W2 = build_strassen_2x2_2x2()
            for r in range(7):
                for idx in range(4):
                    U[idx][r] = U2[idx][r]
                for idx in range(4):
                    j = idx // 2
                    k = idx % 2
                    V[j*3 + k][r] = V2[idx][r]
                for idx in range(4):
                    # W2 is indexed by k*2+i for k=0,1 and i=0,1
                    # In 2x2x3, W is indexed by k*2+i for k=0,1,2 and i=0,1
                    # So for k=0,1: W[k*2+i] = W2[k*2+i]
                    W[idx][r] = W2[idx][r]
            
            # Last 4 terms: schoolbook for 2x2x1 (column 2 of B)
            for i in range(2):
                for j in range(2):
                    r = 7 + i*2 + j
                    U[i*2 + j][r] = 1
                    V[j*3 + 2][r] = 1
                    W[2*2 + i][r] = 1  # k=2, i in {0,1}, so W[4+i]
            
            return U, V, W
        
        U223, V223, W223 = build_2x2x3()
        
        r_total = r4 + 4 * 11  # 49 + 44 = 93
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # Part 1: 4x4x4 Kronecker (columns 0-3 of B)
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # Part 2: 4x4x3 using four 2x2x3 blocks (columns 4-6 of B)
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * 11
                
                for r_local in range(11):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W

    # Build the rank-93 decomposition
    U, V, W = build_4x4x7_decomposition()
    
    # Verify validity
    if check_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if something went wrong (should not happen)
    r = n * m * p
    U_s = [[0] * r for _ in range(n * m)]
    V_s = [[0] * r for _ in range(m * p)]
    W_s = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m + j][col] = 1
                V_s[j*p + k][col] = 1
                W_s[k*n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
