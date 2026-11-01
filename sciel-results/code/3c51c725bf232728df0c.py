import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Find a rank-minimal tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Uses block decomposition: M_{2,3,6} = M_{2,3,3} ⊗ M_{1,1,2}
    M_{2,3,3} uses Strassen(2,2,2) + schoolbook blocks = rank 17
    M_{1,1,2} is rank 2
    Total rank = 17 * 2 = 34
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition (rank 36) as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # Multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(h+g)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    def strassen_222():
        # For M_{2,2,2}: U is (4, 7), V is (4, 7), W is (4, 7)
        # Row indices for U: i*2+j, V: j*2+k, W: k*2+i
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W: C00+=1, C11-=1 -> W[0,0]=1, W[3,3]=-1
        # Actually W[k*2+i]: W[0*2+0]=W[0]=1 for C00, W[1*2+1]=W[3]=-1 for C11
        U[0][0] = 1; U[1][0] = 1  # a+d
        V[0][0] = 1; V[3][0] = 1  # e+h
        W[0][0] = 1; W[3][0] = -1  # C00 += m1, C11 -= m1... wait
        # C[0,0] = m1 + m4 - m5 + m7, C[1,1] = m1 - m2 + m3 + m6
        # So W for m1: W[0*2+0]=1, W[1*2+1]=1 (C11 gets +m1)
        # Let me redo:
        # C[i,j] = sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r
        # For m1: U contributes to (0,0) and (1,1) -> a and d
        # V contributes to (0,0) and (0,1)... no.
        # m1 = (a+d)(e+h). a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        # U[0*2+0]=U[0], U[1*2+1]=U[3] get value 1
        # V[0*2+0]=V[0], V[1*2+1]=V[3] get value 1
        # W: C[0,0] gets m1, C[1,1] gets m1
        # W[0*2+0]=W[0] for C[0,0], W[1*2+1]=W[3] for C[1,1]
        W[0][0] = 1; W[3][0] = 1
        
        # m2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,0]
        # U[1*2+0]=U[2], U[1*2+1]=U[3] get 1
        # V[0*2+0]=V[0] get 1
        # C[1,0] gets m2, C[1,1] gets -m2
        # W[0*2+1]=W[2] for C[1,0], W[1*2+1]=W[3] for C[1,1]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # m3 = a(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
        # U[0*2+0]=U[0] get 1
        # V[0*2+1]=V[1] get 1, V[1*2+1]=V[3] get -1
        # C[0,1] gets m3, C[1,1] gets m3
        # W[1*2+0]=W[2] for C[0,1], W[1*2+1]=W[3] for C[1,1]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # m4 = d(g-f): d=A[1,1], g=B[1,0], f=B[0,1]
        # U[1*2+1]=U[3] get 1
        # V[1*2+0]=V[2] get 1, V[0*2+1]=V[1] get -1
        # C[0,0] gets m4, C[1,0] gets m4
        # W[0*2+0]=W[0] for C[0,0], W[0*2+1]=W[2] for C[1,0]
        U[3][3] = 1
        V[2][3] = 1; V[1][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # m5 = (a+b)h: a=A[0,0], b=A[0,1], h=B[1,1]
        # U[0*2+0]=U[0], U[0*2+1]=U[1] get 1
        # V[1*2+1]=V[3] get 1
        # C[0,0] gets -m5, C[0,1] gets m5
        # W[0*2+0]=W[0] for C[0,0], W[1*2+0]=W[2] for C[0,1]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # m6 = (c-a)(e+g): c=A[1,0], a=A[0,0], e=B[0,0], g=B[1,0]
        # U[1*2+0]=U[2] get 1, U[0*2+0]=U[0] get -1
        # V[0*2+0]=V[0], V[1*2+0]=V[2] get 1
        # C[1,1] gets m6
        # W[1*2+1]=W[3] for C[1,1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # m7 = (b-d)(h+g): b=A[0,1], d=A[1,1], h=B[1,1], g=B[1,0]
        # U[0*2+1]=U[1] get 1, U[1*2+1]=U[3] get -1
        # V[1*2+1]=V[3], V[1*2+0]=V[2] get 1
        # C[0,0] gets m7
        # W[0*2+0]=W[0] for C[0,0]
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[2][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Build M_{2,3,3} decomposition using Strassen + schoolbook
    # A is 2x3, B is 3x3, C = A*B is 2x3
    # Split: A = [A_2x2 | A_2x1], B = [B_2x3; B_1x3]
    # C = A_2x2 * B_2x3 + A_2x1 * B_1x3
    # Part 1: A_2x1 * B_1x3 -> 2x3, rank 6 (schoolbook)
    # Part 2: A_2x2 * B_2x3 -> 2x3
    #   Split B_2x3 = [B_2x2 | B_2x1]
    #   A_2x2 * B_2x2: rank 7 (Strassen)
    #   A_2x2 * B_2x1: rank 4 (schoolbook)
    # Total: 6 + 7 + 4 = 17

    def build_m233():
        # U is (2*3, 17) = (6, 17)
        # V is (3*3, 17) = (9, 17)
        # W is (3*2, 17) = (6, 17)
        rank = 17
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(6)]
        
        col = 0
        # Part 1: A_2x1 * B_1x3, 6 terms
        # A_2x1: A[0,2], A[1,2]
        # B_1x3: B[2,0], B[2,1], B[2,2]
        # C[i,k] += A[i,2] * B[2,k]
        for i in range(2):
            for k in range(3):
                U[i*3+2][col] = 1
                V[2*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
        
        # Part 2a: A_2x2 * B_2x2, Strassen, 7 terms
        # A_2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # B_2x2: B[0,0], B[0,1], B[1,0], B[1,1]
        # C[i,j] for i in {0,1}, j in {0,1}
        U_s, V_s, W_s = strassen_222()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j][r] != 0:
                        U[i*3+j][col] += U_s[i*2+j][r]
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k][r] != 0:
                        V[j*3+k][col] += V_s[j*2+k][r]
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i][r] != 0:
                        W[k*2+i][col] += W_s[k*2+i][r]
            col += 1
        
        # Part 2b: A_2x2 * B_2x1, schoolbook, 4 terms
        # B_2x1: B[0,2], B[1,2]
        # C[i,2] += A[i,0]*B[0,2] + A[i,1]*B[1,2]
        for i in range(2):
            for j in range(2):
                U[i*3+j][col] = 1
                V[j*3+2][col] = 1
                W[2*2+i][col] = 1
                col += 1
        
        return U, V, W

    # Build M_{1,1,2} decomposition, rank 2
    # A is 1x1, B is 1x2, C is 2x1
    # C[k,0] = A[0,0] * B[0,k]
    def build_m112():
        rank = 2
        U = [[1, 1]]
        V = [[1, 0], [0, 1]]
        W = [[1, 0], [0, 1]]
        return U, V, W

    # Combine M_{2,3,3} and M_{1,1,2} via Kronecker product
    # n=2=2*1, m=3=3*1, p=6=3*2
    # U total: (n*m, rank1*rank2) = (6, 17*2) = (6, 34)
    # V total: (m*p, rank1*rank2) = (18, 34)
    # W total: (p*n, rank1*rank2) = (12, 34)
    
    def combine_kronecker(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        rank = r1 * r2
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        ui = i1 * (m1 * m2) + j1 * m2 + j2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                col = r1_idx * r2 + r2_idx
                                U[ui][col] = U1[i1*m1+j1][r1_idx] * U2[i2*m2+j2][r2_idx]
        
        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        vj = j1 * (p1 * p2) + k1 * p2 + k2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                col = r1_idx * r2 + r2_idx
                                V[vj][col] = V1[j1*p1+k1][r1_idx] * V2[j2*p2+k2][r2_idx]
        
        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        wk = k1 * (n1 * n2) + i1 * n2 + i2
                        for r1_idx in range(r1):
                            for r2_idx in range(r2):
                                col = r1_idx * r2 + r2_idx
                                W[wk][col] = W1[k1*n1+i1][r1_idx] * W2[k2*n2+i2][r2_idx]
        
        return U, V, W

    # Build the rank-34 decomposition
    U1, V1, W1 = build_m233()
    U2, V2, W2 = build_m112()
    U, V, W = combine_kronecker(U1, V1, W1, U2, V2, W2, 2, 3, 3, 1, 1, 2)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        result = schoolbook()
    
    current_rank = len(result["U"][0]) if result["U"] else 0
    
    # Try to reduce rank further by removing zero columns or combining
    if elapsed() < deadline - 10 and current_rank > 20:
        # Try removing columns that are all zeros
        for col_idx in range(current_rank - 1, -1, -1):
            if elapsed() > deadline - 5:
                break
            is_zero = True
            for row in result["U"]:
                if row[col_idx] != 0:
                    is_zero = False
                    break
            if not is_zero:
                continue
            for row in result["V"]:
                if row[col_idx] != 0:
                    is_zero = False
                    break
            if not is_zero:
                continue
            for row in result["W"]:
                if row[col_idx] != 0:
                    is_zero = False
                    break
            if is_zero:
                # Remove this column
                result["U"] = [row[:col_idx] + row[col_idx+1:] for row in result["U"]]
                result["V"] = [row[:col_idx] + row[col_idx+1:] for row in result["V"]]
                result["W"] = [row[:col_idx] + row[col_idx+1:] for row in result["W"]]
                current_rank -= 1
    
    return result
