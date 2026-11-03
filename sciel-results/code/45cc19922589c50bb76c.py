import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the tensor T of shape (n, m, p)
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Function to verify a decomposition
    def verify(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reconstruct T
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += abs(val - T[i, j, k])
        return error < 1e-9

    # Strassen's 7-term decomposition for 2x2x2
    # M = A * B where A, B are 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # M = [[m11, m12], [m21, m22]]
    # 
    # Strassen's terms:
    # p1 = (a11+a22)*(b11+b22)
    # p2 = (a21+a22)*b11
    # p3 = a11*(b12-b22)
    # p4 = a22*(b21-b11)
    # p5 = (a11+a12)*b22
    # p6 = (a21-a11)*(b11+b12)
    # p7 = (a12-a22)*(b21+b22)
    #
    # m11 = p1 + p4 - p5 + p7
    # m12 = p3 + p5
    # m21 = p2 + p4
    # m22 = p1 - p2 + p3 + p6

    def strassen_2x2_terms():
        """
        Returns 7 terms for 2x2x2 Strassen.
        Each term is (u, v, w) where:
        u: length 4 (flattened 2x2 A)
        v: length 4 (flattened 2x2 B)
        w: length 4 (flattened 2x2 M)
        """
        terms = []
        
        # p1 = (a11+a22)*(b11+b22)
        u1 = [1, 0, 0, 1]  # a11 + a22
        v1 = [1, 0, 0, 1]  # b11 + b22
        w1 = [1, 0, 0, 1]  # contributes to m11 and m22
        terms.append((u1, v1, w1))
        
        # p2 = (a21+a22)*b11
        u2 = [0, 0, 1, 1]  # a21 + a22
        v2 = [1, 0, 0, 0]  # b11
        w2 = [0, 0, 1, 0]  # contributes to m21
        terms.append((u2, v2, w2))
        
        # p3 = a11*(b12-b22)
        u3 = [1, 0, 0, 0]  # a11
        v3 = [0, 1, 0, -1]  # b12 - b22
        w3 = [0, 1, 0, 0]  # contributes to m12
        terms.append((u3, v3, w3))
        
        # p4 = a22*(b21-b11)
        u4 = [0, 0, 0, 1]  # a22
        v4 = [0, 0, 1, -1]  # b21 - b11
        w4 = [1, 0, 1, 0]  # contributes to m11 and m21
        terms.append((u4, v4, w4))
        
        # p5 = (a11+a12)*b22
        u5 = [1, 1, 0, 0]  # a11 + a12
        v5 = [0, 0, 0, 1]  # b22
        w5 = [1, 1, 0, 0]  # contributes to m11 and m12
        terms.append((u5, v5, w5))
        
        # p6 = (a21-a11)*(b11+b12)
        u6 = [0, 0, 1, -1]  # a21 - a11
        v6 = [1, 1, 0, 0]  # b11 + b12
        w6 = [0, 0, 0, 1]  # contributes to m22
        terms.append((u6, v6, w6))
        
        # p7 = (a12-a22)*(b21+b22)
        u7 = [0, 1, 0, -1]  # a12 - a22
        v7 = [0, 0, 1, 1]  # b21 + b22
        w7 = [1, 0, 0, 0]  # contributes to m11
        terms.append((u7, v7, w7))
        
        return terms

    # Try to construct a rank-44 decomposition
    # Split 3x4x4 into:
    # - 1x4x4 part (schoolbook rank 16)
    # - 2x4x4 part (use Strassen on 2x2 blocks)
    
    # For 2x4x4:
    # A is 2x4, B is 4x4
    # Split A into [A1, A2] where A1, A2 are 2x2
    # Split B into [[B11, B12], [B21, B22]] where each is 2x2
    # M = A1*B11 + A2*B21 (left 2x2)
    #     A1*B12 + A2*B22 (right 2x2)
    
    # Each 2x2*2x2 can be done with rank 7 (Strassen)
    # So 4 such products = 28 terms
    # Plus 1x4x4 schoolbook = 16 terms
    # Total = 44 terms
    
    def build_rank_44():
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Part 1: 1x4x4 schoolbook (row 0 of A, all of B, col 0 of M)
        # A[0, j] * B[j, k] -> M[0, k]
        for j in range(4):
            for k in range(4):
                u = [0] * (n * m)  # length 12
                v = [0] * (m * p)  # length 16
                w = [0] * (p * n)  # length 12
                
                u[0 * m + j] = 1  # A[0, j]
                v[j * p + k] = 1  # B[j, k]
                w[k * n + 0] = 1  # M[0, k]
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
        
        # Part 2: 2x4x4 with Strassen
        # A rows 1,2 (0-indexed: 1,2)
        # A = [[A1, A2]] where A1 = A[1:3, 0:2], A2 = A[1:3, 2:4]
        # B = [[B11, B12], [B21, B22]]
        # M rows 1,2 (0-indexed: 1,2)
        # M[1:3, 0:2] = A1*B11 + A2*B21
        # M[1:3, 2:4] = A1*B12 + A2*B22
        
        strassen_terms = strassen_2x2_terms()
        
        # Product 1: A1 * B11 -> M[1:3, 0:2]
        # A1 = A[1:3, 0:2], B11 = B[0:2, 0:2], M11 = M[1:3, 0:2]
        for u_s, v_s, w_s in strassen_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # A1 indices: A[1,0]=idx 4, A[1,1]=idx 5, A[2,0]=idx 8, A[2,1]=idx 9
            # u_s = [a11, a12, a21, a22]
            u[4] += u_s[0]  # A[1,0]
            u[5] += u_s[1]  # A[1,1]
            u[8] += u_s[2]  # A[2,0]
            u[9] += u_s[3]  # A[2,1]
            
            # B11 indices: B[0,0]=idx 0, B[0,1]=idx 1, B[1,0]=idx 4, B[1,1]=idx 5
            # v_s = [b11, b12, b21, b22]
            v[0] += v_s[0]  # B[0,0]
            v[1] += v_s[1]  # B[0,1]
            v[4] += v_s[2]  # B[1,0]
            v[5] += v_s[3]  # B[1,1]
            
            # M11 = M[1:3, 0:2]
            # M[1,0]=idx 4, M[1,1]=idx 5, M[2,0]=idx 8, M[2,1]=idx 9
            # w_s = [m11, m12, m21, m22]
            w[4] += w_s[0]  # M[1,0]
            w[5] += w_s[1]  # M[1,1]
            w[8] += w_s[2]  # M[2,0]
            w[9] += w_s[3]  # M[2,1]
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        # Product 2: A2 * B21 -> M[1:3, 0:2]
        # A2 = A[1:3, 2:4], B21 = B[2:4, 0:2], M11 = M[1:3, 0:2]
        for u_s, v_s, w_s in strassen_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # A2 indices: A[1,2]=idx 6, A[1,3]=idx 7, A[2,2]=idx 10, A[2,3]=idx 11
            u[6] += u_s[0]  # A[1,2]
            u[7] += u_s[1]  # A[1,3]
            u[10] += u_s[2]  # A[2,2]
            u[11] += u_s[3]  # A[2,3]
            
            # B21 indices: B[2,0]=idx 8, B[2,1]=idx 9, B[3,0]=idx 12, B[3,1]=idx 13
            v[8] += v_s[0]  # B[2,0]
            v[9] += v_s[1]  # B[2,1]
            v[12] += v_s[2]  # B[3,0]
            v[13] += v_s[3]  # B[3,1]
            
            # M11 = M[1:3, 0:2]
            w[4] += w_s[0]  # M[1,0]
            w[5] += w_s[1]  # M[1,1]
            w[8] += w_s[2]  # M[2,0]
            w[9] += w_s[3]  # M[2,1]
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        # Product 3: A1 * B12 -> M[1:3, 2:4]
        # A1 = A[1:3, 0:2], B12 = B[0:2, 2:4], M12 = M[1:3, 2:4]
        for u_s, v_s, w_s in strassen_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # A1 indices
            u[4] += u_s[0]  # A[1,0]
            u[5] += u_s[1]  # A[1,1]
            u[8] += u_s[2]  # A[2,0]
            u[9] += u_s[3]  # A[2,1]
            
            # B12 indices: B[0,2]=idx 2, B[0,3]=idx 3, B[1,2]=idx 6, B[1,3]=idx 7
            v[2] += v_s[0]  # B[0,2]
            v[3] += v_s[1]  # B[0,3]
            v[6] += v_s[2]  # B[1,2]
            v[7] += v_s[3]  # B[1,3]
            
            # M12 = M[1:3, 2:4]
            # M[1,2]=idx 6, M[1,3]=idx 7, M[2,2]=idx 10, M[2,3]=idx 11
            w[6] += w_s[0]  # M[1,2]
            w[7] += w_s[1]  # M[1,3]
            w[10] += w_s[2]  # M[2,2]
            w[11] += w_s[3]  # M[2,3]
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        # Product 4: A2 * B22 -> M[1:3, 2:4]
        # A2 = A[1:3, 2:4], B22 = B[2:4, 2:4], M12 = M[1:3, 2:4]
        for u_s, v_s, w_s in strassen_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # A2 indices
            u[6] += u_s[0]  # A[1,2]
            u[7] += u_s[1]  # A[1,3]
            u[10] += u_s[2]  # A[2,2]
            u[11] += u_s[3]  # A[2,3]
            
            # B22 indices: B[2,2]=idx 10, B[2,3]=idx 11, B[3,2]=idx 14, B[3,3]=idx 15
            v[10] += v_s[0]  # B[2,2]
            v[11] += v_s[1]  # B[2,3]
            v[14] += v_s[2]  # B[3,2]
            v[15] += v_s[3]  # B[3,3]
            
            # M12 = M[1:3, 2:4]
            w[6] += w_s[0]  # M[1,2]
            w[7] += w_s[1]  # M[1,3]
            w[10] += w_s[2]  # M[2,2]
            w[11] += w_s[3]  # M[2,3]
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        rank = len(U_cols)
        U = np.array(U_cols).T  # shape (n*m, rank)
        V = np.array(V_cols).T  # shape (m*p, rank)
        W = np.array(W_cols).T  # shape (p*n, rank)
        
        return U, V, W, rank

    # Try rank 44 first
    try:
        U, V, W, rank = build_rank_44()
        
        # Verify
        if verify(U, V, W, rank):
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Ensure integer/half-integer (they should already be integers)
            for i in range(len(U_list)):
                for j in range(len(U_list[i])):
                    val = U_list[i][j]
                    if abs(val - round(val * 2)) / 2 < 1e-9:
                        U_list[i][j] = round(val * 2) / 2
                    else:
                        U_list[i][j] = round(val)
            
            for i in range(len(V_list)):
                for j in range(len(V_list[i])):
                    val = V_list[i][j]
                    if abs(val - round(val * 2)) / 2 < 1e-9:
                        V_list[i][j] = round(val * 2) / 2
                    else:
                        V_list[i][j] = round(val)
            
            for i in range(len(W_list)):
                for j in range(len(W_list[i])):
                    val = W_list[i][j]
                    if abs(val - round(val * 2)) / 2 < 1e-9:
                        W_list[i][j] = round(val * 2) / 2
                    else:
                        W_list[i][j] = round(val)
            
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            print("Rank 44 verification failed")
    except Exception as e:
        print(f"Error in rank 44: {e}")
    
    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
