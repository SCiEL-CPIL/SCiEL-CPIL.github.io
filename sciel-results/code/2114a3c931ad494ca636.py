import time
import numpy as np

def construct():
    deadline = time.time() + 110.0

    n, m, p = 2, 3, 5
    rank = 28

    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28

    r = 0

    # Part 1: A[:,2] * B[2,:] (2x1 * 1x5) = 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    # Part 2: A[:,0:2] * B[0:2,4:5] (2x2 * 2x1) = 4 terms (schoolbook)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1

    # Part 3: A[:,0:2] * B[0:2,0:2] (2x2 * 2x2) = 7 terms (Strassen)
    # M1 = (a00+a11)(b00+b11)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[5][r] = 1
    r += 1

    # M2 = (a00-a11)(b00-b11)
    U[0][r] = 1; U[4][r] = -1
    V[0][r] = 1; V[6][r] = -1
    W[0][r] = 1; W[5][r] = -1
    r += 1

    # M3 = (a10+a11)b11
    U[3][r] = 1; U[4][r] = 1
    V[6][r] = 1
    W[2][r] = 1; W[5][r] = 1
    r += 1

    # M4 = a00 b10
    U[0][r] = 1
    V[5][r] = 1
    W[0][r] = 1; W[1][r] = 1
    r += 1

    # M5 = a11 (b01-b10)
    U[4][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[1][r] = 1; W[2][r] = 1
    r += 1

    # M6 = (a00+a10) b00
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[2][r] = 1; W[3][r] = 1
    r += 1

    # M7 = (a10-a00) b01
    U[3][r] = 1; U[0][r] = -1
    V[1][r] = 1
    W[3][r] = 1; W[4][r] = 1
    r += 1

    # Part 4: A[:,0:2] * B[0:2,2:4] (2x2 * 2x2) = 7 terms (Strassen)
    # M1
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[9][r] = 1
    r += 1

    # M2
    U[0][r] = 1; U[4][r] = -1
    V[2][r] = 1; V[8][r] = -1
    W[4][r] = 1; W[9][r] = -1
    r += 1

    # M3
    U[3][r] = 1; U[4][r] = 1
    V[8][r] = 1
    W[6][r] = 1; W[9][r] = 1
    r += 1

    # M4
    U[0][r] = 1
    V[7][r] = 1
    W[4][r] = 1; W[5][r] = 1
    r += 1

    # M5
    U[4][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[5][r] = 1; W[6][r] = 1
    r += 1

    # M6
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[6][r] = 1; W[7][r] = 1
    r += 1

    # M7
    U[3][r] = 1; U[0][r] = -1
    V[3][r] = 1
    W[7][r] = 1; W[8][r] = 1
    r += 1

    assert r == 28, f"Rank mismatch: {r}"

    # Verification with multiple random tests
    if time.time() < deadline:
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        valid = True
        for trial in range(5):
            A = np.random.randint(-5, 5, size=(n, m)).astype(np.int64)
            B = np.random.randint(-5, 5, size=(m, p)).astype(np.int64)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p), dtype=np.int64)
            for i in range(n):
                for k in range(p):
                    total = 0
                    for j in range(m):
                        for r_idx in range(rank):
                            total += U_arr[i * m + j, r_idx] * V_arr[j * p + k, r_idx] * W_arr[k * n + i, r_idx]
                    C_computed[i, k] = total
            
            if not np.array_equal(C_computed, C_expected):
                valid = False
                break
        
        if not valid:
            # Fallback to schoolbook rank 30
            rank30 = n * m * p
            U30 = [[0] * rank30 for _ in range(n * m)]
            V30 = [[0] * rank30 for _ in range(m * p)]
            W30 = [[0] * rank30 for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U30[i * m + j][col] = 1
                        V30[j * p + k][col] = 1
                        W30[k * n + i][col] = 1
                        col += 1
            return {"U": U30, "V": V30, "W": W30}

    return {"U": U, "V": V, "W": W}
