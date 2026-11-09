import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # 6x6 is 3x3 blocks of 2x2
    n3, m3, p3 = 3, 3, 3
    n2, m2, p2 = 2, 2, 2
    
    # 1. Construct 3x3 Schoolbook Factors
    rank3 = n3 * m3 * p3
    U3 = [[0] * rank3 for _ in range(n3 * m3)]
    V3 = [[0] * rank3 for _ in range(m3 * p3)]
    W3 = [[0] * rank3 for _ in range(p3 * n3)]
    
    c3 = 0
    for i3 in range(n3):
        for j3 in range(m3):
            for k3 in range(p3):
                U3[i3 * m3 + j3][c3] = 1
                V3[j3 * p3 + k3][c3] = 1
                W3[k3 * n3 + i3][c3] = 1
                c3 += 1
                
    # 2. Construct 2x2 Strassen Factors
    rank2 = 7
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    
    # m1 = (a11+a22)(b11+b22) -> c11+c22
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # m2 = (a21+a22)b11 -> c21
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # m3 = a11(b12-b22) -> c12+c22
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # m4 = a22(b21-b11) -> c11+c21
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # m5 = (a11+a12)b22 -> c11+c12
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    
    # m6 = (a21-a11)(b11+b12) -> c22
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # m7 = (a12-a22)(b21+b22) -> c11
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # 3. Combine to 6x6
    rank6 = rank3 * rank2
    U6 = [[0] * rank6 for _ in range(n * m)]
    V6 = [[0] * rank6 for _ in range(m * p)]
    W6 = [[0] * rank6 for _ in range(p * n)]
    
    for i in range(n):
        i3 = i // 2
        i2 = i % 2
        for j in range(m):
            j3 = j // 2
            j2 = j % 2
            for k in range(p):
                k3 = k // 2
                k2 = k % 2
                
                # Iterate over all combinations of c3 and c2
                # Actually, it's more efficient to iterate over c3, c2 and fill
                pass # We will do it differently for efficiency

    # Efficient filling
    for c3 in range(rank3):
        for c2 in range(rank2):
            c6 = c3 * rank2 + c2
            
            # Fill U6
            for i3 in range(n3):
                for j3 in range(m3):
                    val3 = U3[i3 * m3 + j3][c3]
                    if val3 == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val2 = U2[i2 * m2 + j2][c2]
                            if val2 == 0: continue
                            i = i3 * 2 + i2
                            j = j3 * 2 + j2
                            U6[i * m + j][c6] += val3 * val2
                            
                            if time.time() > deadline:
                                return {"U": U6, "V": V6, "W": W6}

            # Fill V6
            for j3 in range(m3):
                for k3 in range(p3):
                    val3 = V3[j3 * p3 + k3][c3]
                    if val3 == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val2 = V2[j2 * p2 + k2][c2]
                            if val2 == 0: continue
                            j = j3 * 2 + j2
                            k = k3 * 2 + k2
                            V6[j * p + k][c6] += val3 * val2
                            
                            if time.time() > deadline:
                                return {"U": U6, "V": V6, "W": W6}

            # Fill W6
            for k3 in range(p3):
                for i3 in range(n3):
                    val3 = W3[k3 * n3 + i3][c3]
                    if val3 == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val2 = W2[k2 * n2 + i2][c2]
                            if val2 == 0: continue
                            k = k3 * 2 + k2
                            i = i3 * 2 + i2
                            W6[k * n + i][c6] += val3 * val2
                            
                            if time.time() > deadline:
                                return {"U": U6, "V": V6, "W": W6}

    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
