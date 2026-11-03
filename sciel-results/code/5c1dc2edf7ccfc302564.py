import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    # Partitions
    I1 = [0, 1]
    I2 = [2]
    J1 = [0, 1]
    J2 = [2, 3]
    J3 = [4]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4, 5]
    K4 = [6]

    blocks = []
    # Strassen blocks (2x2x2)
    for i_set in [I1]:
        for j_set in [J1, J2]:
            for k_set in [K1, K2, K3]:
                blocks.append(('strassen', i_set, j_set, k_set))
    
    # Schoolbook blocks
    for k_set in [K1, K2, K3, K4]:
        blocks.append(('school', I1, J3, k_set))
    for k_set in [K1, K2, K3, K4]:
        blocks.append(('school', I2, J1, k_set))
    for k_set in [K1, K2, K3, K4]:
        blocks.append(('school', I2, J2, k_set))
    for k_set in [K1, K2, K3, K4]:
        blocks.append(('school', I2, J3, k_set))

    total_rank = 0
    for b in blocks:
        if b[0] == 'strassen':
            total_rank += 7
        else:
            total_rank += len(b[1]) * len(b[2]) * len(b[3])

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for b in blocks:
        if time.time() > deadline:
            break
            
        if b[0] == 'strassen':
            i_set, j_set, k_set = b[1], b[2], b[3]
            i1, i2 = i_set
            j1, j2 = j_set
            k1, k2 = k_set
            
            # M1
            U[i1*m+j1][col] = 1; V[j1*p+k1][col] = 1; W[k1*n+i1][col] = 1
            U[i2*m+j2][col] = 1; V[j2*p+k2][col] = 1; W[k2*n+i2][col] = 1
            col += 1
            # M2
            U[i2*m+j1][col] = 1; V[j1*p+k1][col] = 1; W[k1*n+i2][col] = 1
            U[i2*m+j2][col] = 1; V[j1*p+k1][col] = 1; W[k1*n+i2][col] = 1
            col += 1
            # M3
            U[i1*m+j1][col] = 1
            V[j1*p+k2][col] = 1; W[k2*n+i1][col] = 1
            V[j2*p+k2][col] = -1; W[k2*n+i2][col] = 1
            col += 1
            # M4
            U[i2*m+j2][col] = 1
            V[j2*p+k2][col] = 1; W[k2*n+i1][col] = 1
            V[j1*p+k1][col] = -1; W[k1*n+i2][col] = 1
            col += 1
            # M5
            U[i1*m+j1][col] = 1; V[j2*p+k2][col] = 1; W[k2*n+i1][col] = 1
            U[i1*m+j2][col] = 1; V[j2*p+k2][col] = 1; W[k2*n+i2][col] = 1
            col += 1
            # M6
            U[i2*m+j1][col] = 1; V[j1*p+k1][col] = 1; W[k1*n+i2][col] = 1
            U[i1*m+j1][col] = -1; V[j1*p+k1][col] = -1; W[k1*n+i2][col] = -1
            U[i2*m+j1][col] = 1; V[j2*p+k2][col] = 1; W[k2*n+i2][col] = 1
            U[i1*m+j1][col] = -1; V[j2*p+k2][col] = -1; W[k2*n+i2][col] = -1
            col += 1
            # M7
            U[i1*m+j2][col] = 1; V[j1*p+k2][col] = 1; W[k1*n+i1][col] = 1
            U[i2*m+j2][col] = -1; V[j1*p+k2][col] = -1; W[k1*n+i1][col] = -1
            U[i1*m+j2][col] = 1; V[j2*p+k2][col] = 1; W[k2*n+i1][col] = 1
            U[i2*m+j2][col] = -1; V[j2*p+k2][col] = -1; W[k2*n+i1][col] = -1
            col += 1

        else:
            i_set, j_set, k_set = b[1], b[2], b[3]
            for i in i_set:
                for j in j_set:
                    for k in k_set:
                        if time.time() > deadline:
                            break
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
