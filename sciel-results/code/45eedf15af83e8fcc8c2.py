def construct():
    n = 10
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            i5 = i % 5
            i2 = i // 5
            j5 = j % 5
            j2 = j // 5
            
            # L1
            u1 = (i5 + j5) % 5
            v1 = (i2 + j2) % 2
            L1[i][j] = 5 * u1 + v1
            
            # L2
            u2 = (i5 + 2 * j5) % 5
            v2 = i2
            L2[i][j] = 5 * u2 + v2
            
    return [L1, L2]
