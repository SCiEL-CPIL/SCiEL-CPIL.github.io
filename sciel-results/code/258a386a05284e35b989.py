"""Rewritten program: Kronecker-product rank reduction for 2x3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6

    # Kronecker product structure: (1,1,2) otimes (2,3,3)
    # (1,1,2): n1=1, m1=1, p1=2, rank r1=2
    # (2,3,3): n2=2, m2=3, p2=3, rank r2=18
    # Product: (2,3,6) with rank r = r1 * r2 = 36

    n1, m1, p1 = 1, 1, 2
    n2, m2, p2 = 2, 3, 3
    r1 = n1 * m1 * p1  # 2
    r2 = n2 * m2 * p2  # 18
    r = r1 * r2  # 36

    # Build U1, V1, W1 for (1,1,2) schoolbook
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    c = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][c] = 1
                V1[j * p1 + k][c] = 1
                W1[k * n1 + i][c] = 1
                c += 1

    # Build U2, V2, W2 for (2,3,3) schoolbook
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    c = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][c] = 1
                V2[j * p2 + k][c] = 1
                W2[k * n2 + i][c] = 1
                c += 1

    # Kronecker product
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: index (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1 * m1 + j1][c1]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2 * m2 + j2][c2]
                            if u2 == 0:
                                continue
                            idx_u = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                            U[idx_u][c] = u1 * u2
            # V: index (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V1[j1 * p1 + k1][c1]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2 * p2 + k2][c2]
                            if v2 == 0:
                                continue
                            idx_v = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                            V[idx_v][c] = v1 * v2
            # W: index (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W1[k1 * n1 + i1][c1]
                    if w1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2 = W2[k2 * n2 + i2][c2]
                            if w2 == 0:
                                continue
                            idx_w = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                            W[idx_w][c] = w1 * w2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
