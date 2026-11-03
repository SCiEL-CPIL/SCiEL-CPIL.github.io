import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    return {"U": U, "V": V, "W": W}

def _build_4x4x2_strassen():
    n, m, p = 4, 4, 2
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4 blocks of 2x2x2
    # A blocks: (br, bc)
    # B blocks: (br, 0) and (br, 1) -> B is 4x2, so B_top is 2x2, B_bot is 2x2
    # C blocks: (br, 0) and (br, 1)
    
    # We map the 4 Strassen decompositions to the 4 A blocks
    # A00 * B_top, A01 * B_bot, A10 * B_top, A11 * B_bot
    # B_top is B[0:2, 0:2], B_bot is B[2:4, 0:2]
    # C_top is C[0:2, 0:2], C_bot is C[2:4, 0:2]
    
    block_map = [
        (0, 0, 0, 0), # A00, B_top, C_top
        (0, 1, 0, 1), # A01, B_bot, C_top
        (1, 0, 1, 0), # A10, B_top, C_bot
        (1, 1, 1, 1), # A11, B_bot, C_bot
    ]
    
    for idx, (ar, ac, br, cr) in enumerate(block_map):
        offset = idx * 7
        for r in range(7):
            rr = offset + r
            # U: A[ar*2+i, ac*2+j]
            for i in range(2):
                for j in range(2):
                    val = U2[i*2+j][r]
                    if val != 0:
                        U[(ar*2+i)*4 + (ac*2+j)][rr] += val
            # V: B[br*2+i, j] where j is 0,1
            for i in range(2):
                for j in range(2):
                    val = V2[i*2+j][r]
                    if val != 0:
                        V[(br*2+i)*2 + j][rr] += val
            # W: C[cr*2+i, j] where j is 0,1
            for i in range(2):
                for j in range(2):
                    val = W2[i*2+j][r]
                    if val != 0:
                        W[j*4 + (cr*2+i)][rr] += val
                        
    return {"U": U, "V": V, "W": W}

def _build_4x4x1_schoolbook():
    n, m, p = 4, 4, 1
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}

def _build_4x4x3():
    n, m, p = 4, 4, 3
    part1 = _build_4x4x2_strassen()
    rank1 = len(part1['U'][0])
    part2 = _build_4x4x1_schoolbook()
    rank2 = len(part2['U'][0])
    
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: Cols 0-1
    for i in range(16):
        for r in range(rank1):
            U[i][r] = part1['U'][i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank1):
                V[i*p + j][r] = part1['V'][i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank1):
                W[i*n + j][r] = part1['W'][i*4+j][r]
                
    # Part 2: Col 2
    offset = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset + r] = part2['U'][i][r]
    for i in range(4):
        for r in range(rank2):
            V[i*p + 2][offset + r] = part2['V'][i][r]
    for i in range(4):
        for r in range(rank2):
            W[2*n + i][offset + r] = part2['W'][i][r]
            
    return {"U": U, "V": V, "W": W}

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
        
    try:
        part1 = _build_4x4x4_strassen()
        part2 = _build_4x4x3()
        
        rank1 = len(part1['U'][0])
        rank2 = len(part2['U'][0])
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(16)]
        V = [[0] * total_rank for _ in range(28)]
        W = [[0] * total_rank for _ in range(28)]
        
        # Part 1: 4x4x4 (Cols 0-3)
        for i in range(16):
            for r in range(rank1):
                U[i][r] = part1['U'][i][r]
        for i in range(4):
            for j in range(4):
                for r in range(rank1):
                    V[i*7 + j][r] = part1['V'][i*4+j][r]
        for i in range(4):
            for j in range(4):
                for r in range(rank1):
                    W[i*4 + j][r] = part1['W'][i*4+j][r]
                    
        # Part 2: 4x4x3 (Cols 4-6)
        offset = rank1
        for i in range(16):
            for r in range(rank2):
                U[i][offset + r] = part2['U'][i][r]
        for i in range(4):
            for j in range(3):
                for r in range(rank2):
                    V[i*7 + (4+j)][offset + r] = part2['V'][i*3+j][r]
        for i in range(3):
            for j in range(4):
                for r in range(rank2):
                    W[(4+i)*4 + j][offset + r] = part2['W'][i*4+j][r]
                    
        result = {"U": U, "V": V, "W": W}
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
        
    return _schoolbook(4, 4, 7)
