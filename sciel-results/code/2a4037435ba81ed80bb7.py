import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'u': [(0, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1), (1, 1, 1)]},
        {'u': [(1, 0, 1), (1, 1, 1)], 'v': [(0, 0, 1)], 'w': [(0, 1, 1)]},
        {'u': [(0, 0, 1)], 'v': [(0, 1, 1), (1, 1, -1)], 'w': [(1, 0, 1), (1, 1, 1)]},
        {'u': [(1, 1, 1)], 'v': [(1, 0, 1), (0, 0, -1)], 'w': [(0, 0, 1), (0, 1, 1)]},
        {'u': [(0, 0, 1), (0, 1, 1)], 'v': [(1, 1, 1)], 'w': [(0, 0, -1), (1, 0, 1)]},
        {'u': [(1, 0, 1), (0, 0, -1)], 'v': [(0, 0, 1), (0, 1, 1)], 'w': [(1, 1, 1)]},
        {'u': [(0, 1, 1), (1, 1, -1)], 'v': [(1, 0, 1), (1, 1, 1)], 'w': [(0, 0, 1)]}
    ]
    
    col = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            for term in strassen_terms:
                if time.time() > deadline:
                    break
                for i, j, val in term['u']:
                    U[i * m + j_off + j][col] += val
                for j, k, val in term['v']:
                    V[j * p + k_off + k][col] += val
                for k, i, val in term['w']:
                    W[k * n + i][col] += val
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
