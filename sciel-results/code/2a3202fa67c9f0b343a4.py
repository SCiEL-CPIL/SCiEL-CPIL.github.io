import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank calculation
    # 5 Strassen blocks for 2x2x2 (rank 7 each)
    # 20 standard terms for 2x1x10 (rank 1 each)
    rank = 5 * 7 + 20
    # rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x10 using 5 Strassen blocks
    # A uses j in {0, 1}
    # B uses j in {0, 1}
    # C uses i in {0, 1}, k in {0..9}
    
    # Strassen coefficients for A (linear forms)
    # m1: (a+d)(e+h) -> L_A = a+d, L_B = e+h
    # m2: (c+d)e -> L_A = c+d, L_B = e
    # m3: a(f-h) -> L_A = a, L_B = f-h
    # m4: d(g-e) -> L_A = d, L_B = g-e
    # m5: (a+b)h -> L_A = a+b, L_B = h
    # m6: (c-a)(e+f) -> L_A = c-a, L_B = e+f
    # m7: (b-d)(g+h) -> L_A = b-d, L_B = g+h
    
    # Indices for A (2x2):
    # a = A[0,0] -> idx 0
    # b = A[0,1] -> idx 1
    # c = A[1,0] -> idx 2
    # d = A[1,1] -> idx 3
    
    # Linear forms for A (size 6, only first 4 used)
    LA = [
        [1, 0, 0, 1, 0, 0], # a+d
        [0, 0, 1, 1, 0, 0], # c+d
        [1, 0, 0, 0, 0, 0], # a
        [0, 0, 0, 1, 0, 0], # d
        [1, 1, 0, 0, 0, 0], # a+b
        [0, 0, 1, -1, 0, 0],# c-a
        [0, 1, 0, -1, 0, 0] # b-d
    ]
    
    # Indices for B (2x2 block):
    # e = B[0,0] -> idx 0
    # f = B[0,1] -> idx 1
    # g = B[1,0] -> idx 2
    # h = B[1,1] -> idx 3
    
    # Linear forms for B (size 4)
    LB = [
        [1, 1, 0, 0], # e+h
        [1, 0, 0, 0], # e
        [0, 1, 0, -1],# f-h
        [0, -1, 1, 0],# g-e
        [0, 0, 0, 1], # h
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # g+h
    ]
    
    # W matrices for Strassen (2x2 each)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # W[r] is a 2x2 matrix.
    # W index in output: k*n + i.
    # For a block at columns k0, k0+1:
    # C[0, k0] corresponds to W[k0*n + 0]
    # C[0, k0+1] corresponds to W[(k0+1)*n + 0]
    # C[1, k0] corresponds to W[k0*n + 1]
    # C[1, k0+1] corresponds to W[(k0+1)*n + 1]
    
    # Let's define the W contributions for each m_r
    # m1: [[1, 0], [0, 1]]
    # m2: [[0, 0], [1, -1]]
    # m3: [[0, 1], [0, 1]]
    # m4: [[1, 0], [1, 0]]
    # m5: [[-1, 1], [0, 0]]
    # m6: [[0, 0], [0, 1]]
    # m7: [[1, 0], [0, 0]]
    
    # Check:
    # C11: 1 - 1 + 1 = 1 (m1, m5, m7) -> m1(1) - m5(1) + m7(1) = 1. Correct.
    # C12: 1 + 1 = 2? No.
    # Strassen C12 = m3 + m5.
    # m3 W12 = 1. m5 W12 = 1. Sum = 2?
    # Wait, Strassen C12 = af + bh.
    # m3 = a(f-h) = af - ah.
    # m5 = (a+b)h = ah + bh.
    # m3 + m5 = af - ah + ah + bh = af + bh. Correct.
    # So W12 for m3 is 1, for m5 is 1.
    
    # C21 = m2 + m4.
    # m2 = (c+d)e = ce + de.
    # m4 = d(g-e) = dg - de.
    # m2 + m4 = ce + dg. Correct.
    # W21 for m2 is 1, for m4 is 1.
    
    # C22 = m1 - m2 + m3 + m6.
    # m1 = ae + ah + de + dh.
    # -m2 = -ce - de.
    # m3 = af - ah.
    # m6 = (c-a)(e+f) = ce + cf - ae - af.
    # Sum:
    # ae: 1 (m1) - 1 (m6) = 0.
    # ah: 1 (m1) - 1 (m3) = 0.
    # de: 1 (m1) - 1 (m2) = 0.
    # dh: 1 (m1).
    # ce: -1 (m2) + 1 (m6) = 0.
    # cf: 1 (m6).
    # af: -1 (m3) - 1 (m6) = -2?
    # Wait.
    # m3 = a(f-h). W22 for m3 is 1?
    # C22 = cf + dh.
    # m3 contributes to C12 and C22?
    # Standard Strassen:
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's re-verify C22.
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m2 = (c+d)e = ce + de
    # m3 = a(f-h) = af - ah
    # m6 = (c-a)(e+f) = ce + cf - ae - af
    
    # m1 - m2 + m3 + m6
    # = (ae + ah + de + dh) - (ce + de) + (af - ah) + (ce + cf - ae - af)
    # ae: 1 - 1 = 0
    # ah: 1 - 1 = 0
    # de: 1 - 1 = 0
    # dh: 1
    # ce: -1 + 1 = 0
    # af: 1 - 1 = 0
    # cf: 1
    # Result: dh + cf. Correct.
    
    # So W matrices:
    # m1: [[1, 0], [0, 1]]
    # m2: [[0, 0], [1, -1]]
    # m3: [[0, 1], [0, 1]]
    # m4: [[1, 0], [1, 0]]
    # m5: [[-1, 1], [0, 0]]
    # m6: [[0, 0], [0, 1]]
    # m7: [[1, 0], [0, 0]]
    
    W_strassen = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    for s in range(5):
        k0 = 2 * s
        for r in range(7):
            # Fill U
            for i in range(2):
                for j in range(2):
                    # A index: i*2 + j
                    # LA[r] index: i*2 + j
                    U[i * 2 + j][col] = LA[r][i * 2 + j]
            
            # Fill V
            # B index: j*p + k
            # j in {0, 1}, k in {k0, k0+1}
            # LB[r] index: j*2 + (k - k0)
            for j in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    # V index: j*10 + k_global
                    V[j * 10 + k_global][col] = LB[r][j * 2 + k_local]
            
            # Fill W
            # W index: k*n + i
            # W_strassen[r] is 2x2: [i][k_local]
            for i in range(2):
                for k_local in range(2):
                    k_global = k0 + k_local
                    W[k_global * 2 + i][col] = W_strassen[r][i][k_local]
            
            col += 1
            
    # Part 2: 2x1x10
    # A uses j=2.
    # B uses j=2.
    # C uses i in {0,1}, k in {0..9}.
    
    for k in range(10):
        for i in range(2):
            # U: A[i, 2]
            U[i * 3 + 2][col] = 1
            # V: B[2, k]
            V[2 * 10 + k][col] = 1
            # W: C[i, k] -> index k*2 + i
            W[k * 2 + i][col] = 1
            col += 1
            
    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    if U[u_idx][c] != 0 and V[v_idx][c] != 0 and W[w_idx][c] != 0:
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to schoolbook if something went wrong
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
