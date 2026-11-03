def construct():
    n, m, p = 2, 3, 10
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 6 x 40
    V = [[0] * rank for _ in range(m * p)]  # 30 x 40
    W = [[0] * rank for _ in range(p * n)]  # 20 x 40
    
    # Strassen coefficients
    strassen_P = [
        (1, 0, 0, 1),   # t1: p00+p11
        (0, 0, 1, 1),   # t2: p10+p11
        (1, 0, 0, 0),   # t3: p00
        (0, 0, 0, 1),   # t4: p11
        (1, 1, 0, 0),   # t5: p00+p01
        (0, 0, 1, -1),  # t6: p10-p00
        (0, 1, 0, -1),  # t7: p01-p11
    ]
    strassen_Q = [
        (1, 0, 0, 1),   # t1: q00+q11
        (1, 0, 0, 0),   # t2: q00
        (0, 1, 0, -1),  # t3: q01-q11
        (-1, 0, 1, 0),  # t4: q10-q00
        (0, 0, 0, 1),   # t5: q11
        (1, 1, 0, 0),   # t6: q00+q01
        (0, 0, 1, 1),   # t7: q10+q11
    ]
    strassen_R = [
        (1, 0, 0, 1),   # t1
        (0, 0, 1, -1),  # t2
        (0, 1, 0, 1),   # t3
        (1, 0, 1, 0),   # t4
        (-1, 1, 0, 0),  # t5
        (0, 0, 0, 1),   # t6
        (1, 0, 0, 0),   # t7
    ]
    
    col = 0
    for a in range(5):
        for t in range(8):
            if t < 7:
                # P-side: 4 entries (p00,p01,p10,p11) → V[jl*10 + (2a+b)]
                for idx, coeff in enumerate(strassen_P[t]):
                    if coeff != 0:
                        b = idx // 2
                        jl = idx % 2
                        V[jl * 10 + (2*a + b)][col] += coeff
                # Q-side: 4 entries (q00,q01,q10,q11) → U[s*3 + j]
                for idx, coeff in enumerate(strassen_Q[t]):
                    if coeff != 0:
                        j = idx // 2
                        s = idx % 2
                        U[s * 3 + j][col] += coeff
                # R-side: 4 entries (r00,r01,r10,r11) → W[(2a+b)*2 + s]
                for idx, coeff in enumerate(strassen_R[t]):
                    if coeff != 0:
                        b = idx // 2
                        s = idx % 2
                        W[(2*a + b) * 2 + s][col] += coeff
            else:
                # Outer product: P_a[:,2] x Q[2,:]
                V[2 * 10 + 2*a][col] += 1
                V[2 * 10 + 2*a + 1][col] += 1
                U[0 * 3 + 2][col] += 1
                U[1 * 3 + 2][col] += 1
                for b in range(2):
                    for s in range(2):
                        W[(2*a + b) * 2 + s][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
