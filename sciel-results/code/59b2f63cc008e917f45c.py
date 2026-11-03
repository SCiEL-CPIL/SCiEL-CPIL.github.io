import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 145

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook_full():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook_full()

    # Strassen-based decomposition for 3x3 * 3x8
    # Split: A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    #        B = [[B11, B12, B13, B14, B15, B16, B17, B18],
    #             [B21, B22, B23, B24, B25, B26, B27, B28],
    #             [B31, B32, B33, B34, B35, B36, B37, B38]]
    #
    # C[0:2, 0:2] = A[0:2,0:2] * B[0:2,0:2]  (2x2 * 2x2, Strassen rank 7)
    # C[0:2, 2:8] = A[0:2,0:2] * B[0:2,2:8] + A[0:2,2] * B[2,2:8]  (2x2*2x6 + 2x1*1x6)
    # C[2, 0:2]   = A[2,0:2] * B[0:2,0:2]    (1x2 * 2x2, rank 4)
    # C[2, 2:8]   = A[2,0:2] * B[0:2,2:8] + A[2,2] * B[2,2:8]     (1x2*2x6 + 1x1*1x6)
    #
    # Total rank: 7 + (7*3 + 2*6) + 4 + (4*3 + 6) = 7 + 33 + 4 + 18 = 62? 
    # Hmm, let me be more careful.
    #
    # Actually for 2x2 * 2xp with p=6: Strassen gives rank 7*3 = 21
    # For 2x1 * 1xp with p=6: rank 2*6 = 12
    # So C[0:2,2:8]: rank 21 + 12 = 33
    # C[2,0:2]: 1x2 * 2x2, rank 4
    # C[2,2:8]: 1x2 * 2x6 + 1x1 * 1x6, rank 12 + 6 = 18
    # Total: 7 + 33 + 4 + 18 = 62
    #
    # But we need to be careful about the exact Strassen terms.
    
    # Let me use a cleaner approach: direct Strassen for 2x2*2x8 block
    # A_top = A[0:2, 0:2] (2x2), B_top = B[0:2, :] (2x8)
    # This is 2x2 * 2x8, Strassen rank = 7 * 4 = 28
    # A[2, 0:2] * B[0:2, :] is 1x2 * 2x8, rank 16
    # A[0:2, 2] * B[2, :] is 2x1 * 1x8, rank 16
    # A[2, 2] * B[2, :] is 1x1 * 1x8, rank 8
    # Total: 28 + 16 + 16 + 8 = 68
    
    if elapsed() < deadline - 5:
        rank_total = 68
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        col = 0
        
        # Block 1: A[0:2,0:2] * B[0:2,:] using Strassen (2x2 * 2x8)
        # Strassen for 2x2 * 2x8:
        # Let A = [[a,b],[c,d]], B = [[e,f,g,h,i,j,k,l],[m,n,o,p,q,r,s,t]] (2x8)
        # C = A*B is 2x8
        # m1 = (a+d)*(e+m)     -> contributes to C[0,0:4] and C[1,4:8]
        # m2 = (c+d)*(e+f+g+h+i+j+k+l-m-n-o-p-q-r-s-t)  # This is getting complex
        #
        # Actually for 2x2 * 2xp, we can apply Strassen column-wise in pairs.
        # Split B into 4 blocks of 2x2: B = [B1, B2, B3, B4]
        # Apply Strassen to each 2x2 * 2x2 block: 7*4 = 28 terms
        
        # Strassen for 2x2 * 2x2:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # m1 = (a+d)(e+f)
        # m2 = (c+d)(e)
        # m3 = (a)(f-h)
        # m4 = (d)(g-e)
        # m5 = (a+b)(h)
        # m6 = (c-a)(e+g)
        # m7 = (b-d)(f+h)
        # C[0,0] = m1 - m4 + m5 - m7
        # C[0,1] = m4 + m7
        # C[1,0] = m2 + m5
        # C[1,1] = m1 - m2 + m3 - m5
        
        # For each 2x2 block of B (columns 2t, 2t+1):
        for t in range(4):
            b_col1 = 2*t      # first column index in B
            b_col2 = 2*t + 1  # second column index in B
            
            # m1 = (a+d)(e+f)
            U[0, col] = 1; U[1, col] = 1  # a+d: A[0,0]+A[1,1]
            V[0*8+b_col1, col] = 1; V[0*8+b_col2, col] = 1  # e+f: B[0,2t]+B[0,2t+1]
            W[(2*t)*3+0, col] = 1; W[(2*t+1)*3+0, col] = 1  # C[2t,0]+C[2t+1,0]
            W[(2*t)*3+1, col] = 1; W[(2*t+1)*3+1, col] = 1  # C[2t,1]+C[2t+1,1]
            col += 1
            
            # m2 = (c+d)(e)
            U[2, col] = 1; U[3, col] = 1  # c+d: A[1,0]+A[1,1]
            V[0*8+b_col1, col] = 1  # e: B[0,2t]
            W[(2*t)*3+1, col] = 1; W[(2*t+1)*3+1, col] = 1  # C[2t,1]+C[2t+1,1]
            col += 1
            
            # m3 = (a)(f-h)
            U[0, col] = 1  # a: A[0,0]
            V[0*8+b_col2, col] = 1; V[1*8+b_col1, col] = -1  # f-h: B[0,2t+1]-B[1,2t]
            W[(2*t)*3+1, col] = 1; W[(2*t+1)*3+1, col] = 1  # C[2t,1]+C[2t+1,1]
            col += 1
            
            # m4 = (d)(g-e)
            U[3, col] = 1  # d: A[1,1]
            V[1*8+b_col1, col] = 1; V[0*8+b_col1, col] = -1  # g-e: B[1,2t]-B[0,2t]
            W[(2*t)*3+0, col] = 1; W[(2*t+1)*3+0, col] = 1  # C[2t,0]+C[2t+1,0]
            col += 1
            
            # m5 = (a+b)(h)
            U[0, col] = 1; U[1, col] = 1  # a+b: A[0,0]+A[0,1]
            V[1*8+b_col2, col] = 1  # h: B[1,2t+1]
            W[(2*t)*3+0, col] = 1; W[(2*t+1)*3+0, col] = 1  # C[2t,0]+C[2t+1,0]
            col += 1
            
            # m6 = (c-a)(e+g)
            U[2, col] = 1; U[0, col] = -1  # c-a: A[1,0]-A[0,0]
            V[0*8+b_col1, col] = 1; V[1*8+b_col1, col] = 1  # e+g: B[0,2t]+B[1,2t]
            W[(2*t)*3+1, col] = 1; W[(2*t+1)*3+1, col] = 1  # C[2t,1]+C[2t+1,1]
            col += 1
            
            # m7 = (b-d)(f+h)
            U[1, col] = 1; U[3, col] = -1  # b-d: A[0,1]-A[1,1]
            V[0*8+b_col2, col] = 1; V[1*8+b_col2, col] = 1  # f+h: B[0,2t+1]+B[1,2t+1]
            W[(2*t)*3+0, col] = 1; W[(2*t+1)*3+0, col] = 1  # C[2t,0]+C[2t+1,0]
            col += 1
        
        # Block 2: A[2,0:2] * B[0:2,:] (1x2 * 2x8, schoolbook rank 16)
        for j in range(2):
            for k in range(8):
                U[6, col] = 1       # A[2,j]
                V[j*8+k, col] = 1   # B[j,k]
                W[k*3+2, col] = 1   # C[k,2]
                col += 1
        
        # Block 3: A[0:2,2] * B[2,:] (2x1 * 1x8, schoolbook rank 16)
        for i in range(2):
            for k in range(8):
                U[i*3+2, col] = 1   # A[i,2]
                V[2*8+k, col] = 1   # B[2,k]
                W[k*3+i, col] = 1   # C[k,i]
                col += 1
        
        # Block 4: A[2,2] * B[2,:] (1x1 * 1x8, rank 8)
        for k in range(8):
            U[8, col] = 1       # A[2,2]
            V[2*8+k, col] = 1   # B[2,k]
            W[k*3+2, col] = 1   # C[k,2]
            col += 1
        
        assert col == rank_total, f"Column mismatch: {col} vs {rank_total}"
        
        if validate(U.tolist(), V.tolist(), W.tolist()):
            best_U, best_V, best_W, best_rank = U.tolist(), V.tolist(), W.tolist(), rank_total
    
    return {"U": best_U, "V": best_V, "W": best_W}
