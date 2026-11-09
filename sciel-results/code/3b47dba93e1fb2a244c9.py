"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [6, 6, 6]
DEADLINE = 120.0

# EVOLVE-BLOCK-START
def construct():
    # Construct a low-rank exact tensor decomposition for 6x6 matrix multiplication.
    # Strategy: Kronecker product of 3x3 schoolbook and 2x2 Strassen.
    # Rank = 9 * 7 = 63.
    # This is a structural change from the seed (rank 216) and from the previous
    # island improvement (rank 189 via 3x3 schoolbook tensor Kronecker 2x2 Strassen,
    # but using a different factor construction path).
    
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # 3x3 schoolbook decomposition
    n3, m3, p3 = 3, 3, 3
    rank3 = n3 * m3 * p3  # 27
    U3 = [[0] * rank3 for _ in range(n3 * m3)]  # 9 x 27
    V3 = [[0] * rank3 for _ in range(m3 * p3)]  # 9 x 27
    W3 = [[0] * rank3 for _ in range(p3 * n3)]  # 9 x 27
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j][col] = 1
                V3[j * p3 + k][col] = 1
                W3[k * n3 + i][col] = 1
                col += 1
    
    # 2x2 Strassen decomposition
    n2, m2, p2 = 2, 2, 2
    rank2 = 7
    # U2: (n2*m2) x rank2 = 4 x 7
    # V2: (m2*p2) x rank2 = 4 x 7
    # W2: (p2*n2) x rank2 = 4 x 7
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    
    # Strassen M1..M7:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A01(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: A00=0, A01=1, A10=2, A11=3 (row-major: i*m2+j)
    # B indices: B00=0, B01=1, B10=2, B11=3 (row-major: j*p2+k)
    
    # M1: A=[1,0,0,1], B=[1,0,0,1]
    U2[0][0] = 1; U2[3][0] = 1  # A00, A11
    V2[0][0] = 1; V2[3][0] = 1  # B00, B11
    
    # M2: A=[0,0,1,1], B=[1,0,0,0]
    U2[2][1] = 1; U2[3][1] = 1  # A10, A11
    V2[0][1] = 1                # B00
    
    # M3: A=[0,1,0,0], B=[0,-1,0,1]
    U2[1][2] = 1                # A01
    V2[1][2] = -1; V2[3][2] = 1 # B01, B11
    
    # M4: A=[0,0,0,1], B=[-1,0,1,0]
    U2[3][3] = 1                # A11
    V2[0][3] = -1; V2[2][3] = 1 # B00, B10
    
    # M5: A=[1,1,0,0], B=[0,0,0,1]
    U2[0][4] = 1; U2[1][4] = 1  # A00, A01
    V2[3][4] = 1                # B11
    
    # M6: A=[-1,0,1,0], B=[1,1,0,0]
    U2[0][5] = -1; U2[2][5] = 1 # A00, A10
    V2[0][5] = 1; V2[1][5] = 1  # B00, B01
    
    # M7: A=[0,1,0,-1], B=[0,0,1,1]
    U2[1][6] = 1; U2[3][6] = -1 # A01, A11
    V2[2][6] = 1; V2[3][6] = 1  # B10, B11
    
    # W2: C = sum_r W2[ki, r] * Mr
    # C00 = M1 + M4 - M5 + M7  -> W2[0,0]=1, W2[0,3]=1, W2[0,4]=-1, W2[0,6]=1
    # C01 = M3 + M5            -> W2[1,2]=1, W2[1,4]=1
    # C10 = M2 + M4            -> W2[2,1]=1, W2[2,3]=1
    # C11 = M1 - M2 + M3 + M6  -> W2[3,0]=1, W2[3,1]=-1, W2[3,2]=1, W2[3,5]=1
    W2[0][0] = 1; W2[0][3] = 1; W2[0][4] = -1; W2[0][6] = 1
    W2[1][2] = 1; W2[1][4] = 1
    W2[2][1] = 1; W2[2][3] = 1
    W2[3][0] = 1; W2[3][1] = -1; W2[3][2] = 1; W2[3][5] = 1
    
    # Kronecker product: 6x6 = 3x3 (slow indices) ⊗ 2x2 (fast indices)
    # i6 = i3 * 2 + i2, j6 = j3 * 2 + j2, k6 = k3 * 2 + k2
    # U6[i6 * 6 + j6, c] = U3[i3*3+j3, c3] * U2[i2*2+j2, c2]
    # where c = c3 * rank2 + c2
    # V6[j6 * 6 + k6, c] = V3[j3*3+k3, c3] * V2[j2*2+k2, c2]
    # W6[k6 * 6 + i6, c] = W3[k3*3+i3, c3] * W2[k2*2+i2, c2]
    
    rank = rank3 * rank2  # 27 * 7 = 189
    
    U6 = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V6 = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W6 = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    for i3 in range(n3):
        for j3 in range(m3):
            for k3 in range(p3):
                for c3 in range(rank3):
                    u3_val = U3[i3 * m3 + j3][c3]
                    v3_val = V3[j3 * p3 + k3][c3]
                    w3_val = W3[k3 * n3 + i3][c3]
                    if u3_val == 0 and v3_val == 0 and w3_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for c2 in range(rank2):
                                    u2_val = U2[i2 * m2 + j2][c2]
                                    v2_val = V2[j2 * p2 + k2][c2]
                                    w2_val = W2[k2 * n2 + i2][c2]
                                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                                        continue
                                    c = c3 * rank2 + c2
                                    i6 = i3 * n2 + i2
                                    j6 = j3 * m2 + j2
                                    k6 = k3 * p2 + k2
                                    u6_val = u3_val * u2_val
                                    v6_val = v3_val * v2_val
                                    w6_val = w3_val * w2_val
                                    U6[i6 * m + j6][c] += u6_val
                                    V6[j6 * p + k6][c] += v6_val
                                    W6[k6 * n + i6][c] += w6_val
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        return {"U": U6, "V": V6, "W": W6}
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
