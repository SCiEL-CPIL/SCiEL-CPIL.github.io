import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def verify(U, V, W, R):
        """
        Verify the tensor identity:
        sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{k',k} * delta_{j',j''}
        for all i,i',j',j'',k,k'
        """
        U_t = np.array(U).reshape(R, n, m)
        V_t = np.array(V).reshape(R, m, p)
        W_t = np.array(W).reshape(R, p, n)

        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for jp in range(m):
                            for jp2 in range(m):
                                target = 1.0 if (i == ip and k == kp and jp == jp2) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U_t[r, ip, jp] * V_t[r, jp2, kp] * W_t[r, k, i]
                                err = abs(val - target)
                                if err > max_err:
                                    max_err = err
        return max_err

    def schoolbook():
        """Standard rank-32 decomposition."""
        R = n * m * p  # 32
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 = 2x2, rank 7.
        U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        Identity: sum_r U[i'*2+j', r] V[j''*2+k', r] W[k*2+i, r] = delta_{i',i} delta_{k',k} delta_{j',j''}
        """
        # Strassen's 7 products:
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A11+A12)B22
        # M3 = A11(B11-B22)
        # M4 = A22(B11+B12)
        # M5 = (A11+A22)B11
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1+M4-M5+M7
        # C12 = M2+M3
        # C21 = M6+M5
        # C22 = M1-M3+M4+M5

        # Let's construct U, V, W for Strassen.
        # U[i*2+j, r] corresponds to coefficient of A[i,j] in product r.
        # V[j*2+k, r] corresponds to coefficient of B[j,k] in product r.
        # W[k*2+i, r] corresponds to coefficient of product r in C[i,k].

        # Product 0 (M1): (A11-A22)(B21+B22)
        # U: A11=1, A22=-1 -> U[0,0]=1, U[3,0]=-1
        # V: B21=1, B22=1 -> V[2,0]=1, V[3,0]=1
        # W: C11+=1 -> W[0,0]=1 (C11 is i=0,k=0, so W[0*2+0,0]=W[0,0])
        #     C22+=1 -> W[1*2+1,0]=W[3,0]=1

        # Product 1 (M2): (A11+A12)B22
        # U: A11=1, A12=1 -> U[0,1]=1, U[1,1]=1
        # V: B22=1 -> V[3,1]=1
        # W: C11+=1 -> W[0,1]=1, C12+=1 -> W[1,1]=1

        # Product 2 (M3): A11(B11-B22)
        # U: A11=1 -> U[0,2]=1
        # V: B11=1, B22=-1 -> V[0,2]=1, V[3,2]=-1
        # W: C12+=1 -> W[1,2]=1, C22+=-1 -> W[3,2]=-1

        # Product 3 (M4): A22(B11+B12)
        # U: A22=1 -> U[3,3]=1
        # V: B11=1, B12=1 -> V[0,3]=1, V[1,3]=1
        # W: C11+=1 -> W[0,3]=1, C22+=1 -> W[3,3]=1

        # Product 4 (M5): (A11+A22)B11
        # U: A11=1, A22=1 -> U[0,4]=1, U[3,4]=1
        # V: B11=1 -> V[0,4]=1
        # W: C11+=-1 -> W[0,4]=-1, C21+=1 -> W[2,4]=1, C22+=1 -> W[3,4]=1

        # Product 5 (M6): (A21-A11)(B11+B12)
        # U: A21=1, A11=-1 -> U[2,5]=1, U[0,5]=-1
        # V: B11=1, B12=1 -> V[0,5]=1, V[1,5]=1
        # W: C21+=1 -> W[2,5]=1

        # Product 6 (M7): (A12-A22)(B21+B22)
        # U: A12=1, A22=-1 -> U[1,6]=1, U[3,6]=-1
        # V: B21=1, B22=1 -> V[2,6]=1, V[3,6]=1
        # W: C11+=1 -> W[0,6]=1, C22+=1 -> W[3,6]=1

        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1
        U[0][0] = 1; U[3][0] = -1
        V[2][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # M2
        U[0][1] = 1; U[1][1] = 1
        V[3][1] = 1
        W[0][1] = 1; W[1][1] = 1

        # M3
        U[0][2] = 1
        V[0][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = -1

        # M4
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = 1
        W[0][3] = 1; W[3][3] = 1

        # M5
        U[0][4] = 1; U[3][4] = 1
        V[0][4] = 1
        W[0][4] = -1; W[2][4] = 1; W[3][4] = 1

        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1

        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def build_strassen_2x4x4():
        """
        Build rank-28 decomposition for 2x4 * 4x4 = 2x4.
        Split A (2x4) into A1 (2x2) and A2 (2x2).
        Split B (4x4) into B11, B12, B21, B22 (each 2x2).
        C1 = A1*B11 + A2*B21
        C2 = A1*B12 + A2*B22

        Each A_i * B_jk is a 2x2 * 2x2 multiplication, rank 7.
        Total rank = 4 * 7 = 28.
        """
        U_s, V_s, W_s = strassen_2x2x2x2()

        R = 28
        U = [[0] * R for _ in range(nm)]   # 8 x 28
        V = [[0] * R for _ in range(mp)]   # 16 x 28
        W = [[0] * R for _ in range(pn)]   # 8 x 28

        # Block 0: A1 * B11 -> contributes to C1 (columns 0-1)
        # A1 is A[0:2, 0:2], so rows 0-3 of U
        # B11 is B[0:2, 0:2], so rows 0-3 of V
        # C1 is C[0:2, 0:2], so rows 0-3 of W
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] += U_s[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[j * p + k][r] += V_s[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[k * n + i][r] += W_s[k * 2 + i][r]

        # Block 1: A2 * B21 -> contributes to C1 (columns 0-1)
        # A2 is A[0:2, 2:4], so rows 4-7 of U (i*m + j where j=2,3)
        # B21 is B[2:4, 0:2], so rows 8-11 of V (j*p + k where j=2,3; k=0,1)
        # C1 is C[0:2, 0:2], so rows 0-3 of W
        for r in range(7):
            rr = r + 7
            for i in range(2):
                for j in range(2):
                    U[i * m + (j + 2)][rr] += U_s[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[(j + 2) * p + k][rr] += V_s[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[k * n + i][rr] += W_s[k * 2 + i][r]

        # Block 2: A1 * B12 -> contributes to C2 (columns 2-3)
        # A1 is A[0:2, 0:2], so rows 0-3 of U
        # B12 is B[0:2, 2:4], so rows 4-7 of V (j*p + k where j=0,1; k=2,3)
        # C2 is C[0:2, 2:4], so rows 4-7 of W (k*n + i where k=2,3)
        for r in range(7):
            rr = r + 14
            for i in range(2):
                for j in range(2):
                    U[i * m + j][rr] += U_s[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[j * p + (k + 2)][rr] += V_s[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[(k + 2) * n + i][rr] += W_s[k * 2 + i][r]

        # Block 3: A2 * B22 -> contributes to C2 (columns 2-3)
        # A2 is A[0:2, 2:4], so rows 4-7 of U
        # B22 is B[2:4, 2:4], so rows 12-15 of V (j*p + k where j=2,3; k=2,3)
        # C2 is C[0:2, 2:4], so rows 4-7 of W
        for r in range(7):
            rr = r + 21
            for i in range(2):
                for j in range(2):
                    U[i * m + (j + 2)][rr] += U_s[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[(j + 2) * p + (k + 2)][rr] += V_s[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[(k + 2) * n + i][rr] += W_s[k * 2 + i][r]

        return {"U": U, "V": V, "W": W}

    # First, verify the Strassen-based rank-28 solution
    result_28 = build_strassen_2x4x4()
    err = verify(result_28["U"], result_28["V"], result_28["W"], 28)
    if err < 1e-9:
        return result_28

    # Fallback to schoolbook rank 32
    result_32 = schoolbook()
    err = verify(result_32["U"], result_32["V"], result_32["W"], 32)
    if err < 1e-9:
        return result_32

    # Should not reach here
    return schoolbook()
# EVOLVE-BLOCK-END
